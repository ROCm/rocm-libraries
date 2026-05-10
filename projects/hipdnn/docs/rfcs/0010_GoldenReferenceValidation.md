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

**[Validation](#generic-test-runner) (C++, every CI run):**
1. Load bundle from disk, extract golden outputs
2. Execute engine under test
3. Compare engine output to golden output — PASS or FAIL

---

## Detailed Design

### Existing Infrastructure

The golden reference infrastructure is already built and working for batchnorm. The table below summarizes each existing component. This RFC **keeps** the bundle format, core loader, Python framework, and test runner pattern. It **adds** auto-discovery (no manual `INSTANTIATE_TEST_SUITE_P`), graph-derived test identity, a golden-first fallback chain, a bundle verifier, data integrity checks, per-engine tolerance overrides, and CI integration. It **changes** the data location (moves to the integration suite) and the default verification mode (golden-first instead of computed-first).

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

Golden data uses the existing format already established by `LoadGraphAndTensors.hpp` and `Graph.save()`. The graph JSON inside the bundle defines the test — operation type, tensor shapes, data types, and all parameters. The folder path is a storage convention for human navigation; the loader does not depend on it. Each test case is a set of files with a shared base name:

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

Test cases are **fully auto-discovered**: the runner recursively scans the golden data directory for `.json` files and registers each as a gtest parameter. No manual `INSTANTIATE_TEST_SUITE_P` is required — dropping a bundle in the directory is sufficient. Golden tests are not special — the runner must respect the same test-filtering mechanism used by all other tests (e.g., a configuration matrix that determines which operations, shapes, or data types to run or skip). No hard-coded workarounds for skipping golden tests.

A **bundle inspection tool** will validate that bundles are well-formed and runnable. It reads bundles, reports tensor metadata and statistics, and can verify that the graph + tensors will load correctly before they reach CI.

---

### Generation Pipeline

![Generation Pipeline](images/generation_pipeline.png)

Golden data is generated by Python scripts in [`reference_data_scripts/`](../../reference_data_scripts/). The reference source is configurable per operation (see [Reference Sources](#reference-sources)); the existing batchnorm generators use PyTorch. Each generator follows the same three-step pattern:

1. **Define** — create graph and input tensors
2. **Compute** — run a reference source (PyTorch, CPU ref, etc.)
3. **Write** — save the bundle (`.json` + `.tensor{uid}.bin`)

To add a new operation, create a node class and generator script following the existing [`generate_batchnorm_reference.py`](../../reference_data_scripts/generate_batchnorm_reference.py) pattern. See [Workflows](#workflows) for the full workflow.

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

Tolerances are **always defined in code**, never stored in the data bundle. Golden tests will initially use the existing fixed per-operation, per-type tolerances in [`TestTolerances.hpp`](../../test_sdk/include/hipdnn_test_sdk/utilities/TestTolerances.hpp) (e.g., batchnorm inference fp32 = `2e-4`). The codebase also has per-operation dynamic tolerance functions in [`DynamicTolerances.hpp`](../../test_sdk/include/hipdnn_test_sdk/utilities/DynamicTolerances.hpp) that compute tolerances based on tensor dimensions — the goal is to wire these into the golden test runner so a single generic test class handles all operations.

Per-engine tolerance overrides (e.g., a configuration file specifying looser tolerances for engines that produce less precise results) must also apply to golden tests. This allows a global acceptable tolerance with per-engine exceptions, avoiding hard-coded workarounds in test code.

**Acceptance criteria**:
- [ ] Golden data bundles contain no tolerance fields
- [ ] Golden tests initially use fixed tolerances from `TestTolerances.hpp`, with dynamic tolerances as the target
- [ ] Per-engine tolerance overrides apply to golden tests the same as all other tests

---

### Data Integrity

Internal consistency is guaranteed by construction: `Graph.save()` writes the JSON and `.bin` files from the same in-memory graph in a single call, so UIDs and tensor data always correspond. `loadGraphAndTensors()` reads UIDs from the JSON and loads the matching `.bin` files. Corruption can only enter after generation — partial downloads, disk errors, or manual edits.

Three checks catch the real failure modes:

1. **Tensor size validation (C++, load time)** — *proposed*. After loading tensor data, verify that each tensor's size matches the dimensions declared in the graph JSON. Catches truncated downloads, wrong-file swaps, and stale files. `loadGraphAndTensors()` does not perform this check today; a truncated file silently produces garbage.

2. **NaN/Inf rejection (Python, generation time)** — *proposed*. `Graph.save()` should reject output tensors containing NaN or Inf before writing any files. Built into `Graph.save()` itself so no generator can bypass it. All-same-value tensors are valid (e.g., a bias-only layer can produce uniform output).

3. **NaN/Inf rejection (bundle verifier, offline)** — *proposed*. Safety net for bundles generated by external tools or before check #2 is added. Run by the CLI verifier before commit, not at test load time — scanning every tensor at runtime adds overhead the test runner should not pay. The existing test validators already catch NaN/Inf from the engine's computation.

A standalone CLI verifier will run these checks across a directory tree before bundles are committed, catching errors before they reach CI.

**Acceptance criteria**:
- [ ] All three checks implemented with actionable error messages naming the tensor UID
- [ ] File size mismatch and NaN/Inf in golden data are hard FAILs, not warnings
- [ ] CLI verifier validates full and graph-only bundles offline

---

## Folder Convention

The recommended folder layout below organizes bundles for human navigation. The loader works with any folder structure — it only requires a valid `.json` file. The convention is a guideline, not enforced by the system.

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

The runner recursively scans the golden data directory for `.json` files. Each `.json` file becomes a separate gtest parameter — one file, one test case. The gtest name is derived from the graph content (operation, layout, data type), not from the folder path.

To add a test case, drop a new `.json` + `.bin` set anywhere under the golden data directory — the next run picks it up automatically. To select tests, use `--gtest_filter` with graph-derived names:

```
--gtest_filter=*BatchnormFwdInference*              # all batchnorm inference tests
--gtest_filter=*BatchnormFwdInference*fp32*          # batchnorm inference fp32 only
--gtest_filter=*ConvFwd*nhwc*fp16*                   # conv forward, nhwc, fp16
--gtest_filter=*Small*                               # all tests named "Small"
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

| CI Stage | Verification Mode | Golden Data Required |
|----------|-------------------|---------------------|
| Pre-submit (smoke) | `auto` | Yes (tests without golden data fall back to computed) |
| Post-submit (full) | `both` | Yes |
| Nightly | `golden` | Yes (tests without golden data are skipped) |

### Workflows

**Add a new golden test** (developer):
1. Write a generation script following the [`generate_batchnorm_reference.py`](../../reference_data_scripts/generate_batchnorm_reference.py) pattern, run it to produce a bundle
2. Run the bundle inspection tool to verify the bundle is well-formed
3. Commit the bundle to the golden data directory (DVC for large tensors). No C++ changes needed — auto-discovery picks it up on the next run

**Debug a customer issue** (support):
1. Receive the customer's bundle (`.json` + `.bin` files) — no source code or NDA required
2. Drop the bundle into the golden data directory
3. Run tests — the runner auto-discovers it, executes the engine, compares against golden output
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
2. **C++ graph export**: The codebase can already export graphs to JSON via `Graph.save()`. The remaining work is a convenience utility that takes an existing test-as-code `buildGraph()`, runs it, and writes the complete bundle — enabling bulk conversion of computed tests to golden tests.
3. **External data validation**: Because bundles are self-contained and tool-agnostic, external parties (customers, partner teams) could submit their own bundles for validation. A Python-only comparison tool could load two bundles with the same graph and diff their output tensors — no C++ required.
4. **Reproducible generation**: Fixed seeds for random input generation so that regenerating a bundle (after a schema change, PyTorch upgrade, or generator fix) produces the same inputs, isolating output differences to the reference source change.
