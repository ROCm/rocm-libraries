# RFC 0010: Golden Reference Validation

> Owner: Integration Test Team
> Last updated: 2026-05-10

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
   - [Tier Folders](#tier-folders)
   - [Recommended Sub-Structure](#recommended-sub-structure)
   - [How Test Discovery Works](#how-test-discovery-works)
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

**Long-term direction**: Every integration test becomes a data bundle — no C++ test code, no `buildGraph()` functions, no per-operation test classes. Adding a test means dropping files in a folder. The `computed` mode and its `buildGraph()` infrastructure remain available during the transition, but the end state is a test suite that is entirely data-driven.

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

The golden reference infrastructure is already built and working for batchnorm. The table below summarizes each existing component. This RFC **keeps** the bundle format, core loader, and Python framework. It **adds** a fully generic test runner (recursive scan, no per-operation C++ code), graph-derived test identity, a golden-first fallback chain, a bundle verifier, data integrity checks, per-engine tolerance overrides, and CI integration. It **replaces** the per-operation `INSTANTIATE_TEST_SUITE_P` discovery with recursive auto-discovery, and **moves** the data location to the integration suite.

| Component | File(s) | Role |
|-----------|---------|------|
| Core loader | [`LoadGraphAndTensors.hpp`](../../test_sdk/include/hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp) | Reads bundles from disk, separates inputs from expected outputs, validates results |
| Ref runner | [`GoldenReferenceCpu.hpp`](../../test_sdk/tests/utilities/GoldenReferenceCpu.hpp) | Base fixture + `getGoldenReferenceParams(subDir)` — scans a directory for `.json` files, each becomes a gtest parameter. Renamed to `ValidateGoldenBundleWithRef` in this RFC |
| Plugin runner | [`GoldenReferenceGpu.hpp`](../../../../dnn-providers/miopen-provider/tests/common/GoldenReferenceGpu.hpp) | Same pattern, executes via engine plugin (defined, no tests yet). Renamed to `ValidateGoldenBundleWithPlugin` in this RFC |
| Test instantiation (current) | [`TestCpuFpReferenceBatchnorm.cpp`](../../test_sdk/tests/utilities/TestCpuFpReferenceBatchnorm.cpp) | One class + `INSTANTIATE_TEST_SUITE_P` per operation/layout/datatype — replaced by the generic runner in this RFC |
| Tolerance defaults | [`TestTolerances.hpp`](../../test_sdk/include/hipdnn_test_sdk/test/TestTolerances.hpp) | Per-operation compile-time atol/rtol constants (e.g., `getToleranceInference<T>()` for batchnorm) |
| TOML overrides | [`TestSettings.hpp`](../../test_sdk/include/hipdnn_test_sdk/test/TestSettings.hpp), engine `.toml` files | Per-engine `[[tolerance_overrides]]` with glob matching — lets specific tests relax tolerances without code changes |
| Dynamic tolerances | [`DynamicTolerances.hpp`](../../test_sdk/include/hipdnn_test_sdk/test/DynamicTolerances.hpp) | Higham-style error bounds computed from tensor dimensions and data type — not yet connected to golden tests |
| Python framework | [`reference_data_scripts/utilities/`](../../reference_data_scripts/utilities/) | Generates bundles: defines graphs, runs PyTorch, writes output files |
| Golden data | [`hipdnn_reference_data/`](../../hipdnn_reference_data/BatchnormFwdInference/) | 6 batchnorm bundles across 4 layout/datatype combinations. Currently flat (no tier folder) — will be moved under `quick/` when the new folder convention is adopted |

#### What this RFC keeps, modifies, and replaces

The table below maps each existing component to its treatment in this RFC.

| Component | Status | What changes |
|-----------|--------|-------------|
| `LoadGraphAndTensors.hpp` | **Kept, extended** | Core loading functions (`loadGraphAndTensors()`, `extractAndClearOutputTensorData()`, `validateTensors()`) are the foundation of the generic test runner. Add tensor size validation after loading (check #2). |
| `TestTolerances.hpp` | **Kept, extended** | Per-operation compile-time defaults remain the starting point. The generic runner adds a lookup-by-operation-type mechanism to select the right defaults from this file at runtime. |
| TOML override infrastructure | **Kept as-is** | `TestSettings.hpp`, `TestConfig.hpp`, and per-engine `.toml` files with `[[tolerance_overrides]]` glob matching are connected to golden tests — a TOML override that matches a golden test name applies automatically. |
| Python generation framework | **Kept, enhanced** | `reference_data_scripts/utilities/` (`graph.py`, `tensor.py`, `common.py`) remain the generation backbone. Enhanced: generator scripts auto-derive output paths from graph content (operation type, layout, data type) to match the folder convention. |
| Existing golden data | **Kept, relocated** | 6 batchnorm bundles move from `hipdnn_reference_data/` to `golden_reference_data/quick/BatchnormFwdInference/...` under the new folder convention. Bundle contents unchanged. |
| `GoldenReferenceCpu.hpp` → `ValidateGoldenBundleWithRef` | **Pattern replaced** | The base fixture pattern (`goldenReferenceTestSuite()`, `getGoldenReferenceParams(subDir)`) is replaced by the generic test runner. Key difference: no per-operation test class, recursive discovery across tier folders, tolerance looked up by operation type instead of hard-coded per fixture. The existing functions inform the generic runner design. |
| `GoldenReferenceGpu.hpp` → `ValidateGoldenBundleWithPlugin` | **Pattern replaced** | Same as ref runner — the per-operation fixture pattern is replaced by a single generic plugin runner class. Currently has no tests; the generic runner provides the first plugin golden tests. |
| `TestCpuFpReferenceBatchnorm.cpp` | **Replaced** | The per-operation test class + `INSTANTIATE_TEST_SUITE_P` pattern is replaced by recursive auto-discovery. Adding a new operation no longer requires writing a C++ test class — drop bundle files in the right tier folder. |
| `DynamicTolerances.hpp` | **Future integration** | Higham-style error bounds are not connected to golden tests in this RFC. Future work: use dynamic tolerances as a fallback when `TestTolerances.hpp` has no entry for an operation type. |

### Self-Contained Bundles

All of these components operate on a single shared artifact -- the golden data bundle. A bundle (`{Name}.json` + `{Name}.tensor{uid}.bin`) is self-contained. The graph JSON carries the full computation definition. The `.bin` files carry the raw tensor data (inputs and outputs). Together they are a complete test case. The bundle does not reference any C++ code, any `buildGraph()` function, or any test fixture. If the computation changes, generate a new bundle.

The bundle format is **deliberately independent of any test infrastructure**. The JSON follows the FlatBuffers `graph.fbs` schema; the `.bin` files are raw contiguous tensors. Any tool that can parse JSON and read binary can consume bundles — they are not tied to GTest, `ValidateGoldenBundleWithRef`, or the `getGoldenReferenceParams()` discovery mechanism. Examples of other consumers:

- **External test harnesses** — a partner's internal test framework loads the same bundles to validate their engine
- **Python validation scripts** — a PyTorch script loads a bundle, re-runs the computation, compares
- **Pre-commit bundle verifier** — the proposed [verifier](#data-integrity) consumes bundles directly for integrity checking

A bundle can be **full** or **graph-only**. A **full bundle** (`{Name}.json` + `.bin` files) is the primary format — it carries pre-computed tensor data and the runner loads it and compares. This is the end-state format: every test is a full bundle with known-good reference data.

A **graph-only bundle** (`{Name}.json` alone, no `.bin` files) is a **transitional tool** for migrating existing computed tests. It carries only the computation definition, no tensor data. The runner generates inputs, runs the engine under test and a reference source, compares their outputs, and optionally writes the resulting `.bin` files back to produce a full bundle. Graph-only bundles let us move test definitions from `buildGraph()` to disk incrementally — once the `.bin` files are generated and committed, the graph-only bundle becomes a full bundle and the migration for that test is complete.

#### Bundle metadata (open for discussion)

A bundle today carries the computation definition and tensor data — everything needed to run the test. It does not carry provenance information: who generated it, when, with what tool version, or from what source model.

Adding an optional `metadata` object to the graph JSON could improve traceability:

```json
{
  "metadata": {
    "generator": "reference_data_scripts/batchnorm_inference.py",
    "generated_at": "2026-05-11T14:30:00Z",
    "pytorch_version": "2.3.0",
    "rocm_version": "6.4.0",
    "source_model": "resnet50",
    "notes": "baseline for RFC 0010 migration"
  }
}
```

This is **not required for the test runner** — the runner ignores fields it doesn't recognize. The metadata is purely for humans debugging a failure: "this bundle was generated 6 months ago with PyTorch 2.1 — has the reference changed since then?"

**Open questions**:
- Which fields (if any) should be standardized vs. freeform?
- Should the generator scripts populate metadata automatically, or is it opt-in?
- Is this valuable enough to include in v1, or should it wait until we see a real need?

This section is intentionally left open for team input.

### Golden Data Format

The bundle format is a convention, not a library API. It uses the existing format already established by `LoadGraphAndTensors.hpp` (C++ reader) and `Graph.save()` (Python writer), but any tool that follows the same convention can produce or consume bundles. A bundle is a **directory** containing a set of files with a shared base name: one `.json` file (graph definition conforming to the [`graph.fbs`](../../flatbuffers_sdk/schemas/graph.fbs) schema) and one `.tensor{uid}.bin` file per tensor (raw contiguous data matching the tensor's declared `data_type`, `dims`, and `strides`).

```
{Name}/                    # One directory per bundle
  {Name}.json              # Graph definition (operation, tensor metadata, parameters)
  {Name}.tensor{uid}.bin   # Raw tensor data, one file per UID
```

Each bundle is a self-contained folder. Share a bundle = zip the folder. Delete a bundle = delete the folder. No risk of mixing files across bundles.

For example, batchnorm inference bundles at `BatchnormFwdInference/nchw/fp32/`:

```
BatchnormFwdInference/nchw/fp32/
  typical/
    typical.json               # Graph: batchnorm inference, 6 tensors, fp32
    typical.tensor0.bin         # x (input)
    typical.tensor1.bin         # mean (input)
    typical.tensor2.bin         # inv_variance (input)
    typical.tensor3.bin         # scale (input)
    typical.tensor4.bin         # bias (input)
    typical.tensor5.bin         # y (output — golden reference)
  large_batch/
    large_batch.json           # Same operation, large batch dimension
    large_batch.tensor0.bin
    ...
  resnet50_layer3/
    resnet50_layer3.json       # Same operation, shape from a real model
    resnet50_layer3.tensor0.bin
    ...
```

Each bundle lives in its own directory. The name describes *why the test exists* — `typical` covers a common shape, `large_batch` stresses the batch dimension, `resnet50_layer3` tests a real-world shape. Tensor dimensions are in the graph JSON, not the name. See [Folder Convention](#folder-convention) for the full directory structure.

#### Binary tensor format

Each `.tensor{uid}.bin` file is a **raw dump of the tensor's underlying storage** — no header, no metadata, no framing. The graph JSON carries all the metadata needed to interpret the bytes:

| Property | Source | Details |
|----------|--------|---------|
| Element type | `data_type` field in JSON | `FLOAT` = 4 bytes, `HALF` = 2 bytes, `BFLOAT16` = 2 bytes, etc. |
| Dimensions | `dims` field in JSON | Element counts per dimension (e.g., `[1, 3, 224, 224]`) |
| Strides | `strides` field in JSON | **Element strides**, not byte strides (e.g., `[150528, 50176, 224, 1]`). Multiply by `sizeof(element_type)` for byte offsets. |
| Byte order | Native platform | Little-endian on x86-64. No endianness marker in the file. |

The file size in bytes equals `element_space × sizeof(element_type)`, where `element_space` is computed from `dims` and `strides` (the total storage footprint including any gaps for non-contiguous layouts). For a contiguous (packed) tensor, `element_space` equals the product of `dims`.

**To read a `.bin` file**: allocate `element_space × sizeof(T)` bytes, read the file into that buffer, then index using the strides from JSON. For contiguous tensors this is a straightforward row-major (C-order) array.

---

### Generic Test Runner

![Validation Pipeline](images/validation_pipeline.png)

The runner is a **single generic test class** — not one class per operation. It does not know what operation a bundle contains until it loads the graph JSON at runtime. Adding a test means dropping files in a folder. No C++ changes, no recompile.

#### How it works

For a **full bundle** (`.json` + `.bin` files):

1. **Discover** — recursively scan the golden data directory for `.json` files. Each file becomes a test case.
2. **Load** — read the bundle from disk, separate golden outputs from inputs
3. **Execute** — run the engine under test (CPU reference or MIOpen GPU plugin)
4. **Compare** — check engine output against golden output — PASS or FAIL

For a **graph-only bundle** (`.json` only, no `.bin` files — transitional): the runner loads the graph, generates inputs (using a fixed seed for reproducibility), runs the engine under test *and* a reference source (e.g., CPU reference for GPU tests), and compares their outputs. This enables incremental migration of existing computed tests to bundles — once the outputs are generated and committed as `.bin` files, the graph-only bundle becomes a full bundle.

Tolerance is looked up at runtime from the graph content: the runner reads the operation type and data type from the JSON, then follows the [tolerance priority chain](#two-questions-two-levels) (TOML override → per-operation default). No per-operation test class needed.

#### What a failure looks like

When a golden test fails, the output should give the developer everything needed to diagnose the problem without re-running or adding instrumentation:

```
FAIL: ValidateGoldenBundleWithRef/ConvFwd_nhwc_fp16_resnet50_layer3
  Bundle: quick/ConvFwd/nhwc/fp16/resnet50_layer3/resnet50_layer3.json
  Tensor: y (UID 8, output)
  Shape:  [1, 64, 56, 56]  fp16
  Max absolute error: 3.72e-03  (tolerance: 1e-03)
  Max relative error: 1.15e-02  (tolerance: 1e-02)
  Worst element: index [0, 17, 33, 41]  expected: 0.2148  actual: 0.2185
  Mismatched elements: 42 / 200704 (0.02%)
```

The key fields: which tensor failed, the worst-case error with its location, and how many elements exceeded tolerance. The existing `validateTensors()` in `LoadGraphAndTensors.hpp` already produces per-tensor comparison results — the generic runner surfaces them in this format.

#### Test discovery

The runner uses one `INSTANTIATE_TEST_SUITE_P` per tier, not per operation. These are fixed — they never change as new operations or bundles are added:

```cpp
// Four fixed instantiations — one per tier. Never changes.
INSTANTIATE_TEST_SUITE_P(, ValidateGoldenBundleWithRef,
    discoverGoldenBundles("quick"));          // smoke tier

INSTANTIATE_TEST_SUITE_P(Standard, ValidateGoldenBundleWithRef,
    discoverGoldenBundles("standard"));

INSTANTIATE_TEST_SUITE_P(Comprehensive, ValidateGoldenBundleWithRef,
    discoverGoldenBundles("comprehensive"));

INSTANTIATE_TEST_SUITE_P(Full, ValidateGoldenBundleWithRef,
    discoverGoldenBundles("full"));
```

`discoverGoldenBundles(tierDir)` recursively scans the tier directory for `.json` files and returns each as a gtest parameter. The test name is derived from the graph content (operation, layout, data type) and the bundle name — not from the folder path. If the tier directory is empty or missing, it returns an empty list (no tests, no failure).

At startup, the runner also scans the golden data root for **unexpected top-level directories** — any directory that is not one of the four tier names (`quick`, `standard`, `comprehensive`, `full`) triggers a warning. This catches tier folder typos (e.g., `quik/` instead of `quick/`) that would otherwise silently leave bundles undiscovered.

#### What changes from today

| Aspect | Current (batchnorm only) | This RFC |
|--------|--------------------------|----------|
| Test class | One per operation/layout/datatype | One generic class for all |
| Discovery | `getGoldenReferenceParams(subDir)` — shallow scan, one call per directory | `discoverGoldenBundles(tierDir)` — recursive scan, one call per tier |
| Adding a test | Drop files (existing op) or write C++ class (new op) | Drop files. Always. |
| Tier assignment | GTest prefix per `INSTANTIATE_TEST_SUITE_P` | Tier folder at top level |
| Tolerance | Hard-coded per test class | Looked up from graph content at runtime |

The **pre-commit bundle verifier** (see [Data Integrity](#data-integrity)) validates that bundles are well-formed and runnable before they reach CI. It reads bundles, runs integrity checks (SHA-256, file size, NaN/Inf), and reports tensor metadata and statistics.

**Acceptance criteria**:
- [ ] Single generic test class handles all operation types
- [ ] Recursive scan discovers all bundles — no per-operation C++ code
- [ ] Adding a new test requires only dropping files in a tier folder
- [ ] Test name derived from graph content, not folder path
- [ ] Unexpected top-level directories in golden data root produce a warning
- [ ] Empty or missing tier directory produces zero tests, not a failure

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
python generate_batchnorm_reference.py --name typical
# → BatchnormFwdInference/nchw/fp32/typical/typical.json + .bin
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
| `golden` | Test-as-data tests (`ValidateGoldenBundleWithRef` / `ValidateGoldenBundleWithPlugin`) — graph loaded from disk; tests without golden data are skipped | Pre-computed data from bundle |
| `both` | Both suites, independently — both must pass | Runtime executor + pre-computed data |

**Floating-point edge case**: `-0.0` vs `+0.0` uses value comparison, not bitwise. NaN handling is covered in [Data Integrity](#data-integrity).

---

### Tolerance Framework

Tolerances are **always defined in code and configuration**, never stored in the data bundle. A bundle says *what* to compute and *what* to expect — the tolerance says *how close is close enough*. Keeping tolerances out of the bundle means the same bundle can be validated with different tolerance policies (strict for a CPU reference engine, looser for a GPU engine that trades precision for throughput) without regenerating data.

#### Two questions, two levels

Tolerance selection answers two independent questions:

1. **"How much error should this operation produce?"** — the per-operation default. This is a property of the *computation* (operation type, data type, tensor dimensions). It is the same regardless of which engine runs the graph.

2. **"Does this engine need more room?"** — the per-engine override. This is a property of the *engine*. A specific engine may legitimately exceed the mathematical bound for certain operations — e.g., a fused kernel trading precision for throughput, or a backend whose accumulation order differs from the reference.

These compose into a two-level priority chain:

| Priority | Source | Depends on | Example |
|----------|--------|------------|---------|
| 1 | TOML per-engine override | Engine | `MIOPEN_ENGINE.toml`: conv fp16 → atol `1e-3` |
| 2 | Per-operation default | Operation + data type | batchnorm inference fp32 → atol `2e-4` |

If a TOML override matches, it wins. Otherwise the per-operation default applies. This chain is permanent — it does not change as the system evolves.

The TOML mechanism already exists. Each engine's config (e.g., [`MIOPEN_ENGINE.toml`](../../../dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml)) can declare `[[tolerance_overrides]]` entries with glob-pattern filters and atol/rtol values. The integration harness queries these via `TestConfig::findToleranceOverride(testName)`.

#### How the per-operation default evolves

The per-operation default (level 2) starts simple and gets smarter:

- **This RFC — fixed tolerances**: [`TestTolerances.hpp`](../../test_sdk/include/hipdnn_test_sdk/utilities/TestTolerances.hpp) defines compile-time constants per operation and data type (e.g., batchnorm inference fp32 = `2e-4`, conv fwd fp32 = `1e-5`). Golden tests will use these same fixed values.

- **Future — dynamic tolerances**: [`DynamicTolerances.hpp`](../../test_sdk/include/hipdnn_test_sdk/utilities/DynamicTolerances.hpp) computes tolerances from tensor dimensions using Higham error bounds. Functions already exist for conv, matmul, batchnorm, layernorm, RMS norm, and pointwise. Dynamic tolerances will replace fixed values as the per-operation default, giving tighter bounds for small tensors and appropriately looser bounds for large ones. Fixed values remain as fallback for operations without a dynamic function.

This evolution changes *how* the default is computed — it does not change the two-level chain. TOML overrides still win at every stage.

#### Current gap

The golden ref framework (`ValidateGoldenBundleWithRef`) takes tolerances as hard-coded function parameters — it bypasses both the TOML override lookup and the operation-based tolerance selection. Golden tests must be connected to the same two-level flow: check TOML override first, fall back to per-operation default.

**Acceptance criteria**:
- [ ] Golden data bundles contain no tolerance fields
- [ ] Golden tests check TOML override first, then fall back to per-operation default
- [ ] Per-engine TOML overrides apply to golden tests — no separate override mechanism

---

### Data Integrity

Internal consistency is guaranteed by construction: `Graph.save()` writes the JSON and `.bin` files from the same in-memory graph in a single call, so UIDs and tensor data always correspond. `loadGraphAndTensors()` reads UIDs from the JSON and loads the matching `.bin` files. Corruption can only enter after generation — partial downloads, disk errors, or manual edits.

Four checks catch the real failure modes:

1. **Per-tensor SHA-256 checksum (Python, generation time; C++, load time)** — *proposed*. `Graph.save()` computes a SHA-256 hash of each `.bin` file and stores it in the tensor's JSON entry. At load time, `loadGraphAndTensors()` recomputes the hash and rejects mismatches. This catches truncated downloads, wrong-file swaps, file corruption, and accidental mixing of `.bin` files from different bundles.

2. **Tensor size validation (C++, load time)** — *proposed*. After loading tensor data, verify that `tensor.size == graph.tensor.size()` — i.e., the loaded byte count matches `element_space × sizeof(element_type)` declared in the graph JSON (see [binary tensor format](#binary-tensor-format)). Catches truncated downloads, wrong-file swaps, and stale files. `loadGraphAndTensors()` does not perform this check today; a truncated file silently produces garbage.

3. **NaN/Inf rejection (Python, generation time)** — *proposed*. `Graph.save()` should reject output tensors containing NaN or Inf before writing any files. Built into `Graph.save()` itself so no generator can bypass it. All-same-value tensors are valid (e.g., a bias-only layer can produce uniform output).

4. **NaN/Inf rejection (pre-commit bundle verifier)** — *proposed*. Safety net for bundles generated by external tools or before check #3 is added. The **pre-commit bundle verifier** scans output tensors and rejects any containing NaN or Inf. This runs before commit, **never at test load time** — scanning every tensor at runtime adds overhead the test runner should not pay. The existing test validators already catch NaN/Inf from the engine's computation at runtime.

The **pre-commit bundle verifier** is a standalone CLI tool that runs checks #1–#4 across a directory tree before bundles are committed, catching errors before they reach CI. It also validates structural conventions:

5. **Tier folder validation** — warn about top-level directories that are not one of the four tier names
6. **Graph JSON validation** — verify each `.json` file is a parseable graph (not a stray `README.json` or editor config)
7. **Missing `.bin` files** — for full bundles, verify that every tensor UID in the graph JSON has a corresponding `.bin` file

**Acceptance criteria**:
- [ ] All checks implemented with actionable error messages naming the file and tensor UID
- [ ] File size mismatch and NaN/Inf in golden data are hard FAILs, not warnings
- [ ] Pre-commit bundle verifier validates full and graph-only bundles before commit
- [ ] Stray non-graph `.json` files and unexpected top-level directories produce warnings

---

## Folder Convention

The top-level directory is organized by **tier**. Below that, the structure is flexible — the runner recursively scans for `.json` files regardless of subfolder depth. The recommended (not enforced) convention below the tier is `{Operation}/{Layout}/{DataType}/`, but any structure works as long as each leaf directory has valid bundles.

```
golden_reference_data/
  {Tier}/                           # required: quick, standard, comprehensive, full
    ... any folder structure ...
      {Name}/                       # one directory per bundle
        {Name}.json + {Name}.tensor{uid}.bin
```

In the source tree, golden data lives with the integration test suite at `dnn-providers/integration-tests/golden_reference_data/`. At build time, CMake copies it to `<exe_dir>/../lib/golden_reference_data` via `file(COPY ...)` at configure time and `install(DIRECTORY ...)` at install time. At runtime, `--golden-data-dir` or `HIPDNN_TEST_GOLDEN_DATA_DIR` can override the path.

### Tier folders

The top-level folder determines the tier. The runner scans each tier directory separately (see [Generic Test Runner](#test-discovery)), mapping to the standard [tier cascade](../../../dnn-providers/integration-tests/README.md#test-tiers):

| Folder | GTest prefix | `ctest -L` |
|--------|-------------|------------|
| `quick/` | (none) | `quick` |
| `standard/` | `Standard` | `standard` |
| `comprehensive/` | `Comprehensive` | `comprehensive` |
| `full/` | `Full` | `full` |

The ctest label uses `quick` for the smoke tier (backlog: rename to `smoke` for consistency). When that rename lands, the folder and `discoverGoldenBundles()` call update together — no bundle changes needed.

### Recommended sub-structure

Below each tier, the recommended convention is `{Operation}/{Layout}/{DataType}/`:

| Level | Convention | Examples |
|-------|-----------|----------|
| Operation | PascalCase, direction suffix | `BatchnormFwdInference`, `ConvFwd`, `ConvBwd`, `MatmulFwd` |
| Layout | Lowercase | `nchw`, `nhwc`, `ncdhw`, `ndhwc` |
| DataType | Lowercase abbreviation | `fp32`, `fp16`, `bfp16` |
| BundleName | lowercase_snake_case — **one directory per bundle**. Name describes *why the test exists* (the scenario), not the tensor shapes. Shapes are in the graph JSON. | `typical/`, `odd_spatial/`, `single_element/`, `resnet50_layer3/` |

This convention is **guidance for humans**, not enforced by the runner. The runner discovers bundles by recursive scan and derives test identity from graph content, not folder paths. Folders can be reorganized without breaking tests.

**Bundle naming principle**: the name answers *"what breaks if this test fails?"* — not *"what are the dimensions?"* A bundle called `odd_spatial` tells you the test covers non-power-of-2 spatial dimensions. A bundle called `Small_32x32` tells you nothing about why it exists. Good names describe the scenario: `typical`, `large_batch`, `misaligned_channels`, `resnet50_layer3`, `single_element`.

### Example

```
golden_reference_data/
  quick/
    BatchnormFwdInference/
      nchw/
        fp32/
          typical/
            typical.json + typical.tensor{0..5}.bin
          odd_spatial/
            odd_spatial.json + odd_spatial.tensor{0..5}.bin
        fp16/
          typical/
            typical.json + typical.tensor{0..5}.bin
        bfp16/
          typical/
            typical.json + typical.tensor{0..5}.bin
      ncdhw/
        fp32/
          typical/
            typical.json + typical.tensor{0..5}.bin
  standard/
    BatchnormFwdInference/
      nchw/
        fp32/
          large_batch/
            large_batch.json + large_batch.tensor{0..5}.bin
    ConvFwd/
      nhwc/
        fp16/
          resnet50_layer3/
            resnet50_layer3.json + resnet50_layer3.tensor{0..8}.bin
```

### How Test Discovery Works

`discoverGoldenBundles(tierDir)` recursively scans the tier directory for `.json` files and returns each as a gtest parameter. Test names are derived from graph content (operation type, layout, data type) plus the bundle base name — not from folder paths.

**Adding a test** at any level — new bundle, new data type, new layout, new operation — means dropping `.json` + `.bin` files into the appropriate tier folder. No C++ changes, no recompile. The next run picks them up automatically.

#### Filtering examples

The test name is derived from graph content, not the file path. Given these bundles:

| File path | Graph content | Generated test name |
|-----------|--------------|---------------------|
| `quick/BatchnormFwdInference/nchw/fp32/typical/typical.json` | batchnorm fwd inference, nchw, fp32 | `ValidateGoldenBundleWithRef/BatchnormFwdInference_nchw_fp32_typical` |
| `quick/BatchnormFwdInference/nchw/fp16/typical/typical.json` | batchnorm fwd inference, nchw, fp16 | `ValidateGoldenBundleWithRef/BatchnormFwdInference_nchw_fp16_typical` |
| `quick/BatchnormFwdInference/nchw/fp32/odd_spatial/odd_spatial.json` | batchnorm fwd inference, nchw, fp32 | `ValidateGoldenBundleWithRef/BatchnormFwdInference_nchw_fp32_odd_spatial` |
| `standard/ConvFwd/nhwc/fp16/resnet50_layer3/resnet50_layer3.json` | conv fwd, nhwc, fp16 | `Standard/ValidateGoldenBundleWithRef/ConvFwd_nhwc_fp16_resnet50_layer3` |
| `quick/customer_issues/CASE-12345/repro/repro.json` | conv fwd, nchw, fp32 | `ValidateGoldenBundleWithRef/ConvFwd_nchw_fp32_repro` |

Note the last row: the customer dropped a bundle in an unusual folder path, but the test name comes from graph content — the folder path doesn't matter.

Filters match the generated test name:

```bash
# All batchnorm inference golden tests (any tier, any data type)
--gtest_filter=*BatchnormFwdInference*

# Batchnorm inference fp32 only
--gtest_filter=*BatchnormFwdInference*fp32*

# All "typical" scenario bundles across all operations
--gtest_filter=*typical*

# All conv fwd golden tests
--gtest_filter=*ConvFwd*

# All bundles testing ResNet50 shapes
--gtest_filter=*resnet50*

# Standard-tier tests only (GTest prefix)
--gtest_filter=Standard/*

# One specific test
--gtest_filter=*BatchnormFwdInference_nchw_fp32_odd_spatial
```

---

## CLI and Configuration

The test binary will accept two flags (neither exists today). `--verification-mode` controls **what** runs. `--golden-data-dir` controls **where** golden data is read from (ignored when mode is `computed`).

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--vm, --verification-mode` | `auto`, `computed`, `golden`, `both` | `auto` | Which verification strategy to use (see [Verification Modes](#verification-modes)) |
| `--gd, --golden-data-dir` | path | `<exe_dir>/../lib/golden_reference_data` | Where to find golden data (only used when mode includes golden tests) |

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

Golden tests follow the same [tier cascade](../../../dnn-providers/integration-tests/README.md#how-tiers-cascade) as all other integration tests. Tiers are determined by the top-level folder (see [Folder Convention](#folder-convention)), mapped to GTest prefixes by the four fixed `INSTANTIATE_TEST_SUITE_P` calls.

| CI Stage | ctest Command | Verification Mode | Notes |
|----------|--------------|-------------------|-------|
| Pre-submit (smoke) | `ctest -L quick` | `auto` | `quick/` tier golden tests (no GTest prefix); tests without golden data fall back to computed |
| Post-submit | `ctest -L standard` | `both` | `quick/` + `standard/` tier golden tests; both suites must pass |
| Nightly | `ctest -L comprehensive` | `golden` | All tiers up to comprehensive; tests without golden data are skipped |
| Weekly | `ctest -L full` | `golden` | All tiers |

### Workflows

**Add a test — any operation, any data type** (developer):
1. Write a generation script following the [`generate_batchnorm_reference.py`](../../reference_data_scripts/generate_batchnorm_reference.py) pattern, run it to produce a bundle
2. Run the pre-commit bundle verifier to verify the bundle is well-formed
3. Drop the bundle folder into the appropriate tier directory (e.g., `golden_reference_data/quick/ConvFwd/nhwc/fp16/MyTest/`) and commit (DVC for large tensors). No C++ changes, no recompile — the generic runner discovers it automatically

**Debug a customer issue** (support):
1. Receive the customer's bundle folder — no source code or NDA required
2. Drop the folder into any tier directory (e.g., `golden_reference_data/quick/customer_issues/CustomerBundle/`)
3. Run tests — the runner picks it up, executes the engine, compares against golden output
4. Inspect the diff: which tensors diverge, by how much, at which indices

**Reproduce a CI failure locally** (developer):
1. Pull the golden data via DVC
2. Run the failing test with `--gtest_filter=*OperationName*DataType*`
3. The bundle is self-contained — no environment-specific setup beyond the engine under test

---

## Data Management

Golden data lives in two places — **source tree** and **runtime**:

- **Source tree**: Golden data lives at `dnn-providers/integration-tests/golden_reference_data/`. The existing batchnorm data currently lives at `projects/hipdnn/hipdnn_reference_data/` and will be moved here. Small bundles are committed directly to git. Larger bundles are stored in **DVC**, which the repo already uses for large binary assets. Golden data can grow large (a single test case with `8x512x64x64` fp32 tensors is ~64 MB), so DVC is the natural fit.

- **Runtime**: CMake copies golden data to `<exe_dir>/../lib/golden_reference_data` via `file(COPY ...)` at configure time and `install(DIRECTORY ...)` at install time. The `--golden-data-dir` CLI flag or `HIPDNN_TEST_GOLDEN_DATA_DIR` env var overrides this location.

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
2. **Engine-as-bundle-producer**: Any engine can take a graph-only bundle, execute the graph, and write the results back as a full bundle — enabling cross-engine comparison without a shared reference source.
3. **Bundle-to-bundle comparison**: A standalone tool that loads two bundles and diffs their output tensors directly, matched by graph content rather than filename.
4. **Reproducible generation**: Fixed seeds for random input generation so that regenerating a bundle produces the same inputs, isolating output differences to the reference source change.
5. **Auto-tier classification**: The generator suggests the appropriate tier folder based on tensor element counts, matching the existing size conventions.
