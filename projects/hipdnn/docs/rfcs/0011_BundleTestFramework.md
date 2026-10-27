# RFC 0011: Bundle Test Framework

> Owner: Integration Test Team
> Last updated: 2026-10-15

- **Status**: In progress
- **Status notes**: The generic bundle runner, template sweeps, `--verification-mode`, the `hipdnn_golden_data_tests` binary, and `verify_golden_bundles.py` shipped (#8228, #8926, #8641, #10416, #11494). Bundle `.meta.json` tolerance, tolerance calibration, the `forward_source` check, the 800 MB budget check, and a pre-commit hook are not built. Code is in `dnn-providers/integration-tests/src/harness/bundle/`.
- **Last verified**: 2026-10-15 against `develop` (2501d0b28cc)

## Table of Contents
1. [Summary](#summary)
2. [Design Overview](#design-overview)
3. [Existing Infrastructure](#existing-infrastructure)
4. [Detailed Design](#detailed-design)
   - [4.1 The Bundle](#41-the-bundle)
     - [Self-Contained Bundles](#self-contained-bundles)
     - [Golden Data Format](#golden-data-format)
     - [Compressed Template Sweeps](#compressed-template-sweeps)
     - [Folder Convention](#folder-convention)
   - [4.2 Generation](#42-generation)
     - [Generation Pipeline](#generation-pipeline)
     - [Forward-Backward Generation Constraint](#forward-backward-generation-constraint)
     - [Reference Sources](#reference-sources)
   - [4.3 Validation](#43-validation)
     - [Generic Test Runner](#generic-test-runner)
     - [Verification Modes](#verification-modes)
     - [Tolerance Framework](#tolerance-framework)
     - [Data Integrity](#data-integrity)
   - [4.4 CLI and Configuration](#44-cli-and-configuration)
5. [Integration](#integration)
   - [CI Integration](#ci-integration)
   - [Workflows](#workflows)
6. [Migration Plan](#migration-plan)
7. [Data Management](#data-management)
8. [Risk Register](#risk-register)
9. [Known Limitations](#known-limitations)
10. [Alternatives Considered](#alternatives-considered)
11. [Future Work](#future-work)

---

## Summary

The integration test suite validates engine outputs by computing references at runtime. This creates several gaps:

1. **Circular dependency risk**: If the reference executor has a bug, both sides produce the same wrong answer and the test passes
2. **Coverage gap**: Operations not implemented in a reference executor cannot be tested
3. **Non-determinism**: GPU reference results can vary across runs, making failure investigation harder
4. **Slowness**: CPU reference execution for large tensors is the bottleneck in full-tier tests

This RFC defines the bundle test framework. A golden data bundle (graph JSON plus optional tensor `.bin` files) is loaded from disk, and the engine output is checked against the golden data or against a reference executor. The RFC covers all operation types, the folder convention, data integrity checks, and CI integration.

**Long-term direction**: Every integration test becomes a data bundle. Adding a test means dropping files in a folder -- no C++ code, no per-operation test classes. The regression suite's end state is entirely data-driven.

---

## Design Overview

The bundle test framework has two pipelines -- [**generation**](#generation-pipeline) (produces bundles) and [**validation**](#generic-test-runner) (consumes them) -- sharing one logical data model: a graph plus optional golden tensor data. The on-disk format supports two bundle kinds. **Single-graph bundles** use the `{Name}.json` + `{Name}.tensor{uid}.bin` layout for one-off/customer cases. **Template-sweep bundles** store one `graph.template.json` plus a `sweep.json` case matrix for many cases that share one graph topology.

**Design principle -- test identity is explicit and stable.** Single-graph bundles use the bundle folder path as the gtest suite name and the graph file stem as the test name. Template-sweep bundles derive the suite from the sweep location and use the sanitized `cases[].id` as the test name. Folder hierarchy is used for discovery and grouping, but compressed bundles do not depend on a unique graph file path per case.

Compressed template sweeps are **git-side compression only**. They collapse repeated graph JSON files and directory clutter into one template plus a sweep table; they do not reduce tensor `.bin` payload, DVC/S3 storage, or CI transfer volume for the same set of golden tensors.

---

## Existing Infrastructure

The framework builds on infrastructure that other tests also use.

### Reused building blocks

- **A tensor file reader.** `tensorFromFileAndAttributes()` in `LoadGraphAndTensors.hpp` reads one `.bin` file into a tensor that the graph describes, and checks the file size.
- **Two reference executors.** `CpuReferenceGraphExecutor` and `GpuReferenceGraphExecutor` compute expected outputs at runtime.
- **Tolerance machinery.**
  - *Per-operation compile-time defaults* (atol/rtol constants per operation + datatype) in `TestTolerances.hpp`.
  - *Per-engine TOML overrides* with glob matching (`[[tolerance_overrides]]`), plus per-tensor validator selection (`[[validator_overrides]]`).
  - *Dynamic Higham-style error bounds* -- functions exist for the major operations, but the bundle harnesses do not use them.
- **A Python generation framework** that writes bundles. The generators in the tree use PyTorch as the reference source.
- **DVC**, which the repo uses for large binary assets.

### What this RFC adds

1. **Two generic harnesses** -- one runs bundles through an engine, one checks golden data against the reference executors. Bundles are discovered recursively at GTest startup; adding an operation requires no new C++. See [Generic Test Runner](#generic-test-runner).
2. **A tolerance lookup chain** for bundle tests: TOML override, then bundle metadata, then the per-operation default. Dynamic tolerances stay future work. See [Tolerance Framework](#tolerance-framework).
3. **Integrity checks** at generation time, at load time, and in a bundle verifier. See [Data Integrity](#data-integrity).
4. **A folder convention** with the same tier cascade as the rest of the integration suite. See [Folder Convention](#folder-convention).
5. **One location for all bundle data**: `dnn-providers/integration-tests/integration-test-bundles/`.
6. **Compressed template sweeps** for shared-topology bundles, so dtype/layout/shape coverage can be added as rows in `sweep.json` instead of as repeated graph JSON files. This reduces review noise and filesystem clutter, not DVC tensor-data size.

### Code map for implementers

The functional changes are described in the relevant Detailed Design subsections; this table is just a pointer to the files. Paths under `src/` are in `dnn-providers/integration-tests/`.

| File | Role |
|------|------|
| [`src/harness/bundle/BundleDiscovery.hpp`](../../../../dnn-providers/integration-tests/src/harness/bundle/BundleDiscovery.hpp) | Walks the data root once, finds single-graph bundles and sweep cases, and derives GTest names |
| [`src/harness/bundle/IntegrationTestBundle.hpp`](../../../../dnn-providers/integration-tests/src/harness/bundle/IntegrationTestBundle.hpp) | Loads one bundle or sweep case: parses or expands the graph, reads metadata, and records where the tensor blobs are |
| [`src/harness/bundle/BundleRegistration.hpp`](../../../../dnn-providers/integration-tests/src/harness/bundle/BundleRegistration.hpp) | Registers one GTest test per loaded bundle with `::testing::RegisterTest` |
| [`src/harness/bundle/IntegrationBundleVerificationHarness.cpp`](../../../../dnn-providers/integration-tests/src/harness/bundle/IntegrationBundleVerificationHarness.cpp) | `hipdnn_integration_tests`: runs a bundle through the engine and compares the output with an oracle |
| [`src/harness/reference-validation/BundleReferenceValidationHarness.cpp`](../../../../dnn-providers/integration-tests/src/harness/reference-validation/BundleReferenceValidationHarness.cpp) | `hipdnn_golden_data_tests`: compares golden data with the CPU and GPU reference executors |
| [`src/harness/BundleMetadata.hpp`](../../../../dnn-providers/integration-tests/src/harness/BundleMetadata.hpp) | Parses `.meta.json` and sweep-case metadata; holds the VRAM and arch guards |
| [`src/harness/tolerance/ToleranceResolver.hpp`](../../../../dnn-providers/integration-tests/src/harness/tolerance/ToleranceResolver.hpp) | Computes the per-operation default tolerance for each output tensor |
| [`LoadGraphAndTensors.hpp`](../../test_sdk/include/hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp) | `tensorFromFileAndAttributes()` reads one `.bin` file ([Data Integrity #1](#data-integrity)) |
| [`TestTolerances.hpp`](../../test_sdk/include/hipdnn_test_sdk/utilities/TestTolerances.hpp) | Per-operation default tolerances |
| [`TestSettings.hpp`](../../../../dnn-providers/integration-tests/src/harness/TestSettings.hpp), engine `.toml` files | `[[tolerance_overrides]]` and `[[validator_overrides]]` for engine output |
| [`DynamicTolerances.hpp`](../../test_sdk/include/hipdnn_test_sdk/utilities/DynamicTolerances.hpp) | Future: the per-operation default, computed from graph properties |
| [`reference-data-scripts/`](../../../../dnn-providers/integration-tests/reference-data-scripts/) | Golden data generators and `verify_golden_bundles.py` |
| [`migration-scripts/`](../../../../dnn-providers/integration-tests/migration-scripts/) | `import_graph.py` and the capture pipeline that place bundles in the tree |

---

## Detailed Design

The detailed design is organized around the data lifecycle:

- **[4.1 The Bundle](#41-the-bundle)** -- the substrate that everything else operates on
- **[4.2 Generation](#42-generation)** -- how bundles are produced
- **[4.3 Validation](#43-validation)** -- how bundles are consumed
- **[4.4 CLI and Configuration](#44-cli-and-configuration)** -- how runtime behavior is controlled

---

### 4.1 The Bundle

#### Self-Contained Bundles

Every component in this design -- loader, runners, generators, verifier -- operates on a single shared logical artifact: the **golden data bundle**. A bundle does not reference any C++ code or any test fixture. If the computation topology changes, generate a new bundle or template topology.

The bundle format is independent of any test infrastructure. Graph JSON follows the FlatBuffers `graph.fbs` schema; tensor `.bin` files are raw contiguous tensors. Any tool that can parse JSON and read binary can produce or consume bundles.

A bundle can be **single-graph** or **template-sweep**. A **single-graph bundle** is the original one-case format: `{Name}.json`, optional tensor `.bin` files, and optional metadata. A **template-sweep bundle** is the compressed format for many same-topology cases: one `graph.template.json`, one `sweep.json`, and per-case golden data under `golden/{CaseId}/`.

Either bundle kind can be **full** or **graph-only**. A **full bundle** carries pre-computed tensor data for comparison. A **graph-only bundle** carries only the computation definition; references are computed at runtime until `.bin` files are generated and committed.

##### Bundle metadata

Provenance and run-guard data is stored in a **separate companion file** (`{Name}.meta.json`), not in the graph JSON. A template-sweep case carries the same object inline as `cases[].metadata`. This keeps the graph schema 1:1 with the hipDNN `graph.fbs` format and avoids polluting the computation definition with test infrastructure details. The loader (`BundleMetadata.hpp`) reads the run guards (`gpu_architecture`, `minimum_vram_mb`), the input-fill settings (`seed`, `inputs`), and `enforcement_level` ([RFC 0015](0015_EngineSupportClaims.md)). All other fields are for humans, tooling, and traceability.

```
{Name}/
  {Name}.json              # Graph definition
  {Name}.tensor{uid}.bin   # Tensor data (fetched by DVC, not in git)
  {Name}.tensors.dvc       # DVC pointer that lists the .bin files
  {Name}.meta.json         # Provenance and run guards
  {Name}.support.json      # Support claims (RFC 0015)
```

**Portable bundle** (CPU/framework reference — runs on any ASIC):

```json
{
  "format_version": 1,
  "generator": "reference-data-scripts/batchnorm_inference.py",
  "generator_version": "1.0.0",
  "generated_at": "2026-05-11T14:30:00Z",
  "rocm_version": "6.4.0",
  "reference_source": "PyTorch 2.3.0",
  "reference_source_hash": "a3f8c2e1",
  "reference_strategy": "precision_uplift",
  "generation_command": "python generate_batchnorm_reference.py -f typical ...",
  "notes": "baseline for RFC 0011 migration",
  "seed": 42,
  "minimum_vram_mb": 8192
}
```

**Arch-specific bundle** (GPU reference — only valid on the generating ASIC):

```json
{
  "format_version": 1,
  "generator": "reference-data-scripts/generate_sdpa_fwd_golden.py",
  "generator_version": "1.0.0",
  "generated_at": "2026-06-01T10:00:00Z",
  "gpu_architecture": "gfx942",
  "rocm_version": "7.2",
  "reference_source": "AITER 0.1.13",
  "reference_source_hash": "b7d4e9f2",
  "reference_strategy": "matched_precision",
  "seed": 42,
  "minimum_vram_mb": 16384
}
```

When `gpu_architecture` is present, the arch guard skips the test if the current device doesn't match. When absent (as in the first example), the data is portable across all ASICs.

The loader requires only `format_version`, and it must be `1`. Metadata that is present but malformed fails the test. When the metadata object is missing:

- A single-graph bundle without golden output blobs loads with empty metadata.
- A single-graph bundle with golden output blobs fails its test, because nothing describes how the data was made.
- A sweep case without `cases[].metadata` is dropped from registration. If it has golden output blobs, it fails instead.

`verify_golden_bundles.py` requires non-empty `generator` and `reference_source` in every metadata object, and requires `cases[].metadata` in every sweep case. `reference_strategy` and the remaining fields are optional. `import_graph.py`, `--capture-bundles`, and the SDPA generators write metadata; `generate_batchnorm_reference.py` does not. Backward bundles also need a `forward_source` field, which is not yet built (see [Forward-Backward Generation Constraint](#forward-backward-generation-constraint)).

**Field reference:**

| Field | Required | Type | Purpose |
|-------|----------|------|---------|
| `format_version` | Yes | integer | Schema version. Must be `1`. Reader rejects files with missing or unsupported versions to allow future schema evolution. |
| `generator` | Yes (verifier) | string | Path to the script that produced this bundle. |
| `generator_version` | No | string | Version of the generator script. |
| `generated_at` | No | string | ISO 8601 timestamp of generation. |
| `gpu_architecture` | No | string | GPU arch that produced the reference data (e.g. `"gfx942"`). When present, the arch guard skips the test if the base arch of the current device doesn't match. When absent, the data is assumed portable across all ASICs. Only set this when the reference source is a GPU-specific tool whose numerical output varies by architecture. |
| `rocm_version` | No | string | ROCm version used during generation. |
| `reference_source` | Yes (verifier) | string | What computed the reference output (e.g. `"PyTorch 2.3.0"`, `"AITER 0.1.13"`, `"CpuReferenceGraphExecutor"`). Informational — used for traceability, not for guard logic. |
| `reference_source_hash` | No | string | Commit hash or checksum of the reference source for traceability. |
| `reference_strategy` | No | string | How reference precision was chosen (see table below). |
| `operation` | No | string | Operation name. Informational. |
| `generation_command` | No | string | Full command line used to generate the bundle. |
| `notes` | No | string | Free-text notes for humans. |
| `seed` | No | integer | RNG seed used for input generation. The harness also uses it to fill inputs that have no `.bin` file. |
| `inputs` | No | object | Per-input fill settings, keyed by tensor UID as a string (`"3"`). A key that is not an integer fails the test. |
| `minimum_vram_mb` | No | integer | Minimum device VRAM in MB required to run this bundle. The runner skips the test if the device has less. |
| `enforcement_level` | No | string | `applicability`, `buildable`, or `full` (default). How far the engine must get for the test to pass; see [RFC 0015](0015_EngineSupportClaims.md). Any other value fails the test. |

Unknown fields are ignored by the reader, so generators may include additional fields without breaking compatibility.

`reference_strategy` records how the reference precision was chosen (see [Generation Pipeline](#generation-pipeline)):

| Value | When to use |
|-------|-------------|
| `precision_uplift` | Shallow operations (batchnorm, pointwise). Reference computed in higher precision than target, cast down. |
| `matched_precision` | Deep operations (SDPA backward). Reference computed with the same intermediate precision as the DUT's accumulation contract. |

#### Golden Data Format

A single-graph bundle is a **directory** containing files with a shared base name: one `.json` file (graph definition conforming to the [`graph.fbs`](../../flatbuffers_sdk/schemas/graph.fbs) schema) and one `.tensor{uid}.bin` file per tensor (raw contiguous data matching the tensor's declared `data_type`, `dims`, and `strides`). A `{Name}.tensors.dvc` pointer lists the `.bin` files: git tracks the pointer, and DVC stores the data. `tensorFromFileAndAttributes()` in [`LoadGraphAndTensors.hpp`](../../test_sdk/include/hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp) (C++ reader) and [`Graph.save()`](../../../../dnn-providers/integration-tests/reference-data-scripts/utilities/graph.py) (Python writer) implement this format.

```
{Name}/                    # One directory per bundle
  {Name}.json              # Graph definition (operation, tensor metadata, parameters)
  {Name}.tensor{uid}.bin   # Raw tensor data, one file per UID
```

For example, the `quick/` batchnorm inference bundle:

```
BatchnormFwdInference/nchw/fp32/
  Small/
    Small.json               # Graph: batchnorm inference, 6 tensors, fp32
    Small.meta.json
    Small.support.json
    Small.tensors.dvc        # Lists the six .bin files
    Small.tensor0.bin        # x (input)
    Small.tensor1.bin        # mean (input)
    Small.tensor2.bin        # inv_variance (input)
    Small.tensor3.bin        # scale (input)
    Small.tensor4.bin        # bias (input)
    Small.tensor5.bin        # y (output -- golden reference)
```

The name describes *why the test exists* -- `Small` and `Large` are size classes, and `Gqa` or `SmallStats` name the feature under test. Tensor dimensions are in the graph JSON, not the name. See [Folder Convention](#folder-convention) below for the full directory structure.

##### Binary tensor format

Each `.tensor{uid}.bin` file is a **raw dump of the tensor's underlying storage** -- no header, no metadata, no framing. The graph JSON carries all the metadata needed to interpret the bytes:

| Property | Source | Details |
|----------|--------|---------|
| Element type | `data_type` field in JSON | `float` = 4 bytes, `half` = 2 bytes, `bfloat16` = 2 bytes, etc. |
| Dimensions | `dims` field in JSON | Element counts per dimension (e.g., `[1, 3, 224, 224]`) |
| Strides | `strides` field in JSON | **Element strides**, not byte strides (e.g., `[150528, 50176, 224, 1]`). Multiply by `sizeof(element_type)` for byte offsets. |
| Byte order | Native platform | Little-endian on x86-64. No endianness marker in the file. |

The file size in bytes equals `element_space x sizeof(element_type)`, where `element_space` is computed from `dims` and `strides` (the total storage footprint including any gaps for non-contiguous layouts). For a contiguous (packed) tensor, `element_space` equals the product of `dims`. A ragged tensor ([RFC 0014](0014_RaggedTensors.md)) stores only its valid rows, so its element count comes from the last value of its offset tensor.

**To read a `.bin` file**: allocate `element_space x sizeof(T)` bytes, read the file into that buffer, then index using the strides from JSON. For contiguous tensors this is a straightforward row-major (C-order) array.

**v1 scope: dense (packed) tensors only.** The binary format supports strided layouts but not sparse representations (CSR, COO, etc.). Sparse tensor support is deferred to a future revision.

#### Compressed Template Sweeps

Use a **template-sweep bundle** when several cases share the same graph topology and differ only by case data such as tensor dtype, dims, strides, layout fields, scalar attributes, metadata, or golden data. The template is required; one-off graphs and customer drops should stay in the single-graph format.

```
integration-test-bundles/{Tier}/{Operation}/{TopologyName}/
  graph.template.json
  sweep.json
  support.json
  golden/{CaseId}/tensors.dvc
```

`TopologyName` is the directory name for one invariant graph skeleton. It does not have to describe the skeleton. The migration tools (`import_graph.py`, `place_bundles.py`) use a preferred name (`Default`, or for `import_graph.py` a name from the inferred layout) and then the next free `Variant2`, `Variant3`, …. A hand-made sweep can use a descriptive name, for example `Inference` or `Binary`.

`graph.template.json` is the invariant graph JSON topology with placeholders for per-case values. Placeholder syntax is `${case.<field>}`. Scalar placeholders resolve from `cases[].values`, tensor placeholders resolve from `cases[].values.tensors` by tensor UID, and node-attribute placeholders resolve from `cases[].values.attributes` as `${case.attributes.<field>}`. Attribute placeholders are allowed for values such as SDPA mask bounds and diagonal alignment.

The same topology means node count, edge wiring, tensor UID identity, tensor set, and operation sequence stay fixed. Rank changes are allowed in the same topology when they are represented only by per-case `dims` and `strides`. Changing nodes, edges, tensor UID identity, tensor count, or operation sequence creates a new `TopologyName`. A case can change tensor fields that are not placeholders with `tensor_patches`, for example to make a tensor runtime pass-by-value (see [`tensor_patches`](../../../../dnn-providers/integration-tests/docs/file-formats.md#tensor_patches)). v1 intentionally does not support per-case tensor opt-in/opt-out; SDPA stats output, FP8 descale tensors, and GROUP mode sequence-length tensors use separate topologies. If future graphs show enough duplication across those adjacent topologies to make this split painful, the template and sweep schema can be extended with structural conditionals or sweep composition so those cases can be combined deliberately instead of adding that complexity up front.

`sweep.json` owns the case matrix. Excerpt from `quick/BatchnormFwdInference/Inference/sweep.json` (the real case lists all six tensors):

```json
{
  "version": 1,
  "cases": [
    {
      "id": "small_fp32_nchw",
      "values": {
        "io_data_type": "float",
        "tensors": [
          { "uid": 0, "data_type": "float", "dims": [2, 3, 4, 5], "strides": [60, 20, 5, 1] },
          { "uid": 1, "data_type": "float", "dims": [1, 3, 1, 1], "strides": [3, 1, 1, 1] }
        ]
      },
      "golden": { "path": "golden/small_fp32_nchw/tensors.dvc" },
      "metadata": {
        "format_version": 1,
        "operation": "BatchnormFwdInference",
        "generator": "generate_batchnorm_fwd_golden.py",
        "reference_source": "PyTorch 2.12.0+rocm7.2",
        "seed": 42
      }
    }
  ]
}
```

`cases[].id` is stable and unique within the sweep. It is the tooling handle, golden lookup key, failure-output case id, and gtest test name after sanitization. All cases from one sweep register in the same suite and differ by id only.

Case ids must be lowercase_snake_case (`[a-z0-9_]+`); the verifier rejects other ids. Ids should include the discriminator tokens developers need for filtering: a shape or scenario token plus dtype and layout when either varies in the sweep, and feature tokens such as `causal`, `stats`, `fp8`, or `group` when those features vary within the topology. The migration tools build ids from the tokens that vary in the sweep (shape, dtype, layout, and up to three node attributes) and add a six-character content hash only when those tokens are not unique, for example `3_16_128_bfp16_ncl_f4ebfc`. Do not rely on directory names to make dtype/layout filters work for template sweeps.

Golden tensor data is stored under a per-case directory: `golden/{CaseId}/tensors.dvc`. When DVC pulls the tensor blobs, UID-based tensor filenames remain inside that case directory and cannot collide with another case that uses the same tensor UIDs. Sweep-aware loading must key tensor lookup on the logical `CaseId` or `golden.path` and read from `golden/{CaseId}/`; it must not derive `.bin` paths from the `graph.template.json` sibling directory.

Validation rules for template sweeps. `verify_golden_bundles.py` reports all of them. At load time, discovery rejects duplicate ids, and the loader drops other invalid cases from registration.

- duplicate `cases[].id` values are errors;
- case ids that are not lowercase_snake_case are errors;
- missing placeholder values are errors;
- unused `values` entries are warnings so authors can keep descriptive fields;
- duplicate tensor `uid` values within one case are errors;
- tensor `uid` values not present in the template graph are errors;
- cases that omit a template tensor UID, add a new tensor UID, or otherwise change the template tensor set are errors;
- template tensor placeholders for `dims`, `strides`, or `data_type` without matching case tensor entries are errors;
- template node-attribute placeholders without matching `cases[].values.attributes` entries are errors;
- a non-null `golden` without a string `golden.path` is an error, and `golden.path` must name a `tensors.dvc` file;
- a case without `metadata` is an error;
- every expanded case must build and validate the same flatbuffer graph as a single-graph bundle before it can be registered.

#### Folder Convention

Discovery has no fixed folder schema. `discoverBundles()` walks the whole data root once. Every graph `.json` becomes a single-graph test, except companion files (`*.meta.json`, `*.support.json`) and files under a sweep root. Every directory that holds both `graph.template.json` and `sweep.json` becomes a sweep. The first folder below the data root is the tier. The convention below is kept by the tools and checked by the verifier; registration does not enforce it.

Single-graph bundles use `{Operation}/{Layout}/{DataType}/[{Variant}/...]/{BundleName}/`. Some operations use variant levels, for example `standard/SdpaFwd/bhsd/bf16/hd128_causal_batch/Gqa/`. Template sweeps use `{Operation}/{TopologyName}/` because layout, data type, dims, and strides are case data in `sweep.json`.

```
integration-test-bundles/
  {Tier}/                           # quick, standard, comprehensive, full
    {Operation}/{Layout}/{DataType}/[{Variant}/...]/{BundleName}/
      {BundleName}.json
      {BundleName}.meta.json
      {BundleName}.support.json
      {BundleName}.tensors.dvc
    {Operation}/{TopologyName}/
      graph.template.json
      sweep.json
      support.json
      golden/{CaseId}/tensors.dvc
```

In the source tree, bundle data lives with the integration test suite at `dnn-providers/integration-tests/integration-test-bundles/`. CMake copies this folder to `<build>/lib/integration-test-bundles/` at configure time and installs it. The default path is `<exe_dir>/../lib/integration-test-bundles/`, and `--golden-data-dir` or `HIPDNN_TEST_GOLDEN_DATA_DIR` can override it.

> [!NOTE]
> The copy happens only at configure time. After you add bundles or run `dvc pull`, re-run CMake, or the tests do not see the change.

##### Tier folders

The top-level folder determines the tier. It becomes the first part of the GTest suite name (see [Test discovery](#test-discovery)), and the tier YAML files match that prefix. The tiers map to the standard [tier cascade](../../../../dnn-providers/integration-tests/docs/running-tests.md#test-tiers). A graph under any other top-level folder registers under that folder name and matches no tier.

| Folder | GTest suite prefix | `ctest -L` |
|--------|-------------|------------|
| `quick/` | `quick_` | `quick` |
| `standard/` | `standard_` | `standard` |
| `comprehensive/` | `comprehensive_` | `comprehensive` |
| `full/` | `full_` | `full` |

Note: the ctest label uses `quick` for the smoke tier.

##### Recommended sub-structure

For single-graph bundles:

| Level | Convention | Examples |
|-------|-----------|----------|
| Operation | PascalCase operation name, usually with a direction suffix | `BatchnormFwdInference`, `ConvolutionFwd`, `SdpaFwd`, `SdpaFwdRuntimeScale` |
| Layout | Lowercase | `nchw`, `ncdhw`, `bhsd`, `bshd` |
| DataType | Lowercase abbreviation | `fp32`, `fp16`, `bfp16`, `bf16`, `fp8` |
| Variant (optional, repeatable) | lowercase_snake_case feature set | `hd128_causal_batch/`, `hd128_nomask_group/` |
| BundleName | One directory per one-off bundle. Name describes why the test exists, not only tensor shapes. | `Small/`, `Large/`, `Gqa/`, `SmallStats/` |

For template-sweep bundles:

| Level | Convention | Examples |
|-------|-----------|----------|
| Operation | PascalCase operation name, usually with a direction suffix | `BatchnormFwdInference`, `ConvolutionFwd`, `SdpaFwd` |
| TopologyName | Directory name for one invariant graph skeleton. The tools use `Default`, then `Variant2`, `Variant3`, …. | `Default/`, `Variant2/`, `Inference/`, `Binary/` |
| CaseId | lowercase_snake_case in `sweep.json`; becomes the gtest test name after sanitization. Holds the tokens that vary in the sweep so filters remain stable. | `small_fp32_nchw`, `small_fp16_nchw`, `3_16_128_bfp16_ncl_f4ebfc` |

This convention is guidance for humans. A directory is a sweep only when it holds both `graph.template.json` and `sweep.json`; the verifier reports a directory that holds only one of them. The verifier also expects each single-graph file at `<BundleName>/<BundleName>.json` and warns about any other JSON file.

##### Example layout

```
integration-test-bundles/
  quick/
    BatchnormFwdInference/
      Inference/
        graph.template.json
        sweep.json
        support.json
        golden/small_fp32_nchw/tensors.dvc
        golden/small_fp16_nchw/tensors.dvc
        golden/small_fp32_ncdhw/tensors.dvc
        golden/small_bfp16_nchw/tensors.dvc
      nchw/
        fp32/
          Small/
            Small.json + Small.meta.json + Small.support.json + Small.tensors.dvc
    ConvolutionFwd/
      Default/
        graph.template.json
        sweep.json
        support.json
    SdpaFwd/
      Default/
        graph.template.json
        sweep.json
        support.json
      bhsd/
        fp16/
          hd128_causal_mha/
            Small/
              Small.json + Small.meta.json + Small.tensors.dvc
```

##### Filtering by generated test name

The suite name is the bundle folder path relative to the data root, sanitized and joined with `_`. The test name is the graph `.json` stem for a single-graph bundle, and the sanitized `cases[].id` for a template sweep.

| Bundle source | Generated GTest name |
|---------------|---------------------|
| `quick/BatchnormFwdInference/nchw/fp32/Small/Small.json` | `quick_BatchnormFwdInference_nchw_fp32_Small.Small` |
| `quick/BatchnormFwdInference/Inference/sweep.json`, case `small_fp32_nchw` | `quick_BatchnormFwdInference_Inference.small_fp32_nchw` |
| `quick/BatchnormFwdInference/Inference/sweep.json`, case `small_fp16_nchw` | `quick_BatchnormFwdInference_Inference.small_fp16_nchw` |
| `standard/SdpaFwd/Default/sweep.json`, case `{CaseId}` | `standard_SdpaFwd_Default.{CaseId}` |
| `quick/customer_issues/CASE-12345/repro/repro.json` | `quick_customer_issues_CASE_12345_repro.repro` |

**Collision handling**: Single-graph bundles collide when they produce the same suite and test name. Template-sweep bundles collide when two cases in the same sweep use the same `id`, or when two registered cases produce the same sanitized suite/test pair. Discovery rejects collisions with an error naming both sources.

Because template-sweep suite names do not include dtype, layout, dims, or feature flags, `cases[].id` is the stable filtering surface for those fields. Generators must emit ids that preserve the useful filter tokens; reviewers should treat missing dtype/layout/feature tokens as a test discoverability bug.

Filters match the generated test name:

```bash
# All batchnorm inference tests, single-graph or sweep
--gtest_filter=*BatchnormFwdInference*

# Batchnorm inference fp32 sweep rows
--gtest_filter=*BatchnormFwdInference*fp32*

# All single-graph bundles named "Small" across all operations
--gtest_filter=*.Small

# All cases of the default SDPA forward sweep
--gtest_filter=*SdpaFwd_Default*

# All convolution forward tests
--gtest_filter=*ConvolutionFwd*

# All bfp16 tests
--gtest_filter=*bfp16*

# Standard-tier tests only (suite prefix)
--gtest_filter=standard_*

# One specific template-sweep case
--gtest_filter=quick_BatchnormFwdInference_Inference.small_fp32_nchw

# One specific single-graph bundle
--gtest_filter=quick_BatchnormFwdInference_nchw_fp32_Small.Small
```

---

### 4.2 Generation

#### Generation Pipeline

Golden data is generated by Python scripts in [`reference-data-scripts/`](../../../../dnn-providers/integration-tests/reference-data-scripts/). The reference source is configurable per operation (see [Reference Sources](#reference-sources)); the generators in the tree use PyTorch. Each generator follows the same three-step pattern:

1. **Define** -- create graph and input tensors
2. **Compute** -- run the reference source using a **per-operation precision strategy** and cast outputs to the target type. The strategy depends on the operation's rounding depth:
   - **Precision uplift** (shallow operations -- batchnorm, pointwise, activations): compute in higher precision than the target (e.g., fp64 for fp32 bundles). Few rounding events, so the trajectory divergence between reference and DUT is minimal.
   - **Matched-precision** (deep operations -- SDPA backward, multi-step chains): compute with the same intermediate precision as the DUT's accumulation contract (e.g., BF16 inputs with FP32 intermediates for a32 accumulator kernels). Higher-precision references diverge from the DUT's rounding trajectory at every stage, producing deltas dominated by accumulated precision loss rather than kernel bugs -- forcing artificially loose tolerances that hide real regressions.
3. **Write** -- save the bundle (`.json` + `.tensor{uid}.bin`) in the target precision

Each generator takes the output path and base name with `--base-filename`:

| Script | Output |
|--------|--------|
| `generate_batchnorm_reference.py` | Batchnorm inference `.json` and `.bin` files |
| `generate_sdpa_fwd_golden.py` | SDPA forward `.json`, `.bin`, and `.meta.json` files |
| `generate_sdpa_runtime_scale_test_bundle.py` | Graph-only SDPA forward bundle with a runtime scale tensor: `.json` and `.meta.json` |

Golden data exists for batchnorm inference and SDPA forward. All other bundles are graph-only. Most of them are captured from C++ graph tests with `hipdnn_integration_tests --capture-bundles <dir>` and placed in the tree with `migration-scripts/import_graph.py` (see [Add a bundle](../../../../dnn-providers/integration-tests/docs/adding-tests.md#add-a-bundle)).

##### Auto-derived output path

Not yet built. The generator should **auto-derive the output path** from the graph content. The developer supplies only the **bundle name**; the generator computes the directory from the graph's operation type, layout, and data type:

```bash
python generate_batchnorm_reference.py --name typical
# -> BatchnormFwdInference/nchw/fp32/typical/typical.json + .bin
```

#### Forward-Backward Generation Constraint

Not yet built. No generator writes backward bundles, and no check compares backward bundles with forward bundles.

Backward operations (SDPA backward, batchnorm backward, conv backward) consume **saved tensors from the forward pass** as inputs -- `O` and `LSE` for SDPA, `saved_mean` and `inv_variance` for batchnorm, `mask` for dropout. These are not arbitrary tensors: they must be the actual outputs of a mathematically consistent forward computation. Filling them with random values produces a backward bundle whose expected outputs are numerically meaningless.

**Rule**: The generation script must produce forward and backward bundles in a **single pipeline run**:

1. Generate inputs (Q, K, V) from seed
2. Run forward reference → O, LSE
3. Save forward bundle: inputs = [Q, K, V], expected outputs = [O, LSE]
4. Run backward reference using the O, LSE from step 2 → dQ, dK, dV
5. Save backward bundle: inputs = [Q, K, V, O, LSE, dO], expected outputs = [dQ, dK, dV]

At test time, each bundle is standalone -- backward does not run or reference forward. The coupling exists only at generation time.

**Why this matters**: If the forward bundle is regenerated (e.g., PyTorch upgrade changes softmax rounding) without regenerating the backward bundle, the backward test still passes -- it is internally consistent with its own stale inputs. But the forward and backward bundles now reflect different reference implementations. This silent inconsistency is undetectable without metadata.

**Enforcement**: Backward bundles must include a `forward_source` field in `meta.json`:

```json
{
  "forward_source": {
    "method": "pytorch_math_backend",
    "version": "2.3.0",
    "generated_at": "2026-05-20T14:30:00Z"
  }
}
```

A CI check compares the backward bundle's `forward_source` against the forward bundle's metadata. A mismatch (different method, version, or generation timestamp) fails the check before tests run, surfacing stale backward bundles.

#### Reference Sources

The golden data format is **reference-source-agnostic**. Any tool that produces a valid bundle (graph JSON matching the [`graph.fbs`](../../flatbuffers_sdk/schemas/graph.fbs) schema + corresponding `.bin` files) is a valid reference source.

| Category | Examples | Portability |
|----------|----------|-------------|
| Python frameworks | PyTorch, TensorFlow, JAX | Architecture-independent -- bundles valid on any GPU |
| In-house CPU reference | `CpuReferenceGraphExecutor` | Architecture-independent |
| In-house GPU reference | `GpuReferenceGraphExecutor` | **Architecture-dependent** -- bundles tied to the GPU that generated them |
| AMD internal tools | AITER, AOTriton | Architecture-dependent |

**Architecture portability**: GPU-generated golden data may produce different numerical results across architectures, but with appropriate reference precision strategy and calibrated tolerances, cross-architecture use is possible for many operations. Architecture-independent sources (PyTorch, CPU reference) are preferred for portable regression suites. The `gpu_architecture` metadata field tags arch-specific data, and the harness skips it on other architectures where numerical differences exceed tolerance.

---

### 4.3 Validation

#### Generic Test Runner

Two binaries consume bundles. Both use the same discovery (`discoverBundles()`) and the same loader (`IntegrationTestBundle.hpp`). Neither knows what operation a bundle contains until it loads the graph JSON at runtime.

| Binary | Harness | What it checks |
|--------|---------|----------------|
| `hipdnn_integration_tests` | `IntegrationBundleVerificationHarness` | Runs each bundle through the engine under test and compares the output with the oracle that the [verification mode](#verification-modes) selects |
| `hipdnn_golden_data_tests` | `BundleReferenceValidationHarness` | Compares the golden data of each bundle with the CPU and GPU reference executors. Suites end in `_CpuRef` or `_GpuRef`. No engine runs. |

`hipdnn_golden_data_tests` registers a test only when the bundle has golden data and every node in its graph is in the required-op set of that reference (`ReferenceOpCoverage.hpp`). It uses the per-operation default tolerance only. See [`hipdnn_golden_data_tests`](../../../../dnn-providers/integration-tests/docs/running-tests.md#hipdnn_golden_data_tests).

##### How it works

For a **full single-graph bundle** (`.json` + `.bin` files), `hipdnn_integration_tests` does these steps:

1. **Discover** -- at GTest startup, `discoverBundles(bundleDir)` walks the data root once and finds single-graph `.json` files. Each graph path becomes one logical bundle case -- no C++ code per test.
2. **Load** -- before registration, `loadIntegrationTestBundle()` parses the graph JSON, builds the graph flatbuffer, reads the metadata, and records where the `.bin` files are. The test reads the tensor data when it runs.
3. **Guard** -- the test skips when the device has less VRAM than `minimum_vram_mb`, or when its architecture does not match `gpu_architecture`.
4. **Execute** -- run the graph through the engine under test. If the engine declines the graph, the test skips, or fails when a support claim promised support ([RFC 0015](0015_EngineSupportClaims.md)).
5. **Compare** -- check engine output against golden output from the `.bin` files using the [tolerance chain](#three-levels) -- PASS or FAIL

For a **full template-sweep bundle** (`graph.template.json` + `sweep.json` + per-case golden data): discovery registers one logical bundle case per sweep row. The loader expands the template for that case and builds and validates the same flatbuffer graph as a single-graph bundle. It then reads golden tensor data from the directory of `golden.path` (`golden/{CaseId}/`). The loader must not use the template file's sibling path for tensors, because every case reuses the same graph template and tensor UIDs.

For a **graph-only bundle** (single graph or sweep case without golden `.bin` files): the harness fills the inputs (from the input `.bin` files when they exist, otherwise from `seed` and `inputs` in the metadata), runs the engine under test and a reference executor, and compares their outputs at runtime.

Tolerance is looked up at runtime following the [priority chain](#three-levels). No per-operation test class needed.

> [!NOTE]
> A bundle that fails to load (malformed graph JSON, an invalid sweep case, or a sweep case without metadata) is dropped from registration with an `ERROR` log line. The run stays green with fewer tests. Malformed metadata, golden `.bin` files without metadata, and an unreadable or wrong-size `.bin` file fail as registered tests.

##### What a failure looks like

When a bundle test fails, the output gives the developer everything needed to diagnose the problem without re-running or adding instrumentation. The harness writes one report per mismatched output tensor:

```
Comparison FAILED
  Bundle: <data root>/quick/ConvolutionFwd/Default/sweep.json#<CaseId>
  Tensor: <name> (UID 2, output)
  Shape:  [1, 16, 16, 16]  BFLOAT16
  Tolerance: atol=0.1 rtol=0.1
  Tensor diff for "<name>":
    Total elements: 4096
    Mismatched:     42 (1.03%)
    Max abs diff:   3.720000e-01 at [0, 3, 7, 9]
    Mean abs diff:  1.150000e-03
    Worst mismatches:
      [0, 3, 7, 9]: ref=2.148000e-01, impl=5.868000e-01, diff=3.720000e-01
```

##### Test discovery

At GTest startup, `discoverBundles(bundleDir)` walks the data root once, discovers single-graph bundles, expands template-sweep bundles into logical cases, and returns one entry per logical bundle case. A `--gtest_filter` drops excluded cases before they are loaded. The [verification mode](#verification-modes) selects the oracle for each test (golden, GPU ref, CPU ref, or auto-fallback). Registration is per data root, not per operation — it never changes as new operations, bundles, or sweep rows are added.

**Decided: [`::testing::RegisterTest`](https://google.github.io/googletest/advanced.html#registering-tests-programmatically).** Each logical bundle case is registered as its own test at static initialization time. This eliminates per-operation C++ registration code and gives direct control over test naming (see [Test naming scheme](#test-naming-scheme)). `RegisterTest` applies to bundle tests only; C++ graph tests use `INSTANTIATE_TEST_SUITE_P`.

If the data root is missing, `hipdnn_integration_tests` prints a warning and registers no bundle tests. If no test runs at all, and an engine is named or the data root exists, the binary exits with `Error: zero tests ran.` The bundle verifier (`verify_golden_bundles.py`) warns about unexpected top-level directories (e.g., `quik/` instead of `quick/`) to catch typos.

##### Test naming scheme

Single-graph and template-sweep bundles use different sources for the final gtest test name:

| Bundle kind | Suite source | Test source | Example |
|-------------|--------------|-------------|---------|
| Single graph | Bundle folder path relative to the data root | Graph `.json` stem | `quick_BatchnormFwdInference_nchw_fp32_Small.Small` |
| Template sweep | Sweep folder path relative to the data root | Sanitized `cases[].id` | `quick_BatchnormFwdInference_Inference.small_fp32_nchw` |

The tier is the first part of the suite name: `standard/` gives `standard_`, and `quick/` gives `quick_`.

**GTest identifier validity**: All components use `[a-zA-Z0-9_]` only. The sanitizer replaces invalid characters with underscores.

**Worked examples**:

| Bundle source | Generated GTest name |
|---------------|---------------------|
| `quick/BatchnormFwdInference/nchw/fp32/Small/Small.json` | `quick_BatchnormFwdInference_nchw_fp32_Small.Small` |
| `quick/BatchnormFwdInference/Inference/sweep.json`, case `small_fp32_nchw` | `quick_BatchnormFwdInference_Inference.small_fp32_nchw` |
| `quick/BatchnormFwdInference/Inference/sweep.json`, case `small_fp16_nchw` | `quick_BatchnormFwdInference_Inference.small_fp16_nchw` |
| `standard/SdpaFwd/Default/sweep.json`, case `{CaseId}` | `standard_SdpaFwd_Default.{CaseId}` |
| `quick/customer_issues/CASE-12345/repro/repro.json` | `quick_customer_issues_CASE_12345_repro.repro` |

**Collision handling**: see the collision rules under [Filtering by generated test name](#filtering-by-generated-test-name) above. Discovery rejects duplicate sanitized suite/test pairs with an error naming both sources. Within one template sweep, duplicate `cases[].id` values are also an error.

**Acceptance criteria**:
- [x] One generic harness handles all operation types
- [x] Recursive scan discovers all single-graph bundles and expands all template-sweep cases -- no per-operation C++ code
- [x] Adding a new test requires only dropping files in a tier folder or appending a sweep case, then re-running CMake
- [x] Suite names derive from the bundle folder or sweep folder path; test names derive from the graph `.json` stem or the sanitized `cases[].id`
- [x] `verify_golden_bundles.py` warns about unexpected top-level directories in the data root
- [x] A missing data root produces zero bundle tests and a warning, not a failure
- [x] Test name collision (two cases producing the same name) is a hard error at discovery time
- [x] A bad bundle (unreadable or wrong-size `.bin` file, malformed metadata, golden data without metadata) fails its own test case — other tests continue running so CI provides full signal
- [x] An unparseable graph `.json` file is dropped from registration with an `ERROR` log line (no test case can be registered)
- [x] No available reference for a bundle in `auto` mode fails the test, indicating a validation gap

#### Verification Modes

The default verification strategy is **golden-first with automatic fallback**. For each test, the runner selects the best available reference source in order:

1. **Golden data** -- if the bundle has golden data, use it
2. **GPU reference** -- if no golden data but a GPU reference executor is available, use it
3. **CPU reference** -- last resort, if the operation is implemented

The `--verification-mode` flag overrides this default. In every mode, the **engine** is the thing being tested -- the reference source provides the expected output.

| Mode | Reference source | When reference unavailable |
|------|-----------------|---------------------------|
| `auto` (default) | Per-test fallback: golden → GPU ref → CPU ref | FAIL -- indicates a validation gap |
| `golden` | Pre-computed data from bundle | FAIL -- bundle has no golden data |
| `gpu` | GPU reference executor at runtime | FAIL -- GPU ref cannot verify this bundle |
| `cpu` | CPU reference executor at runtime | FAIL -- CPU ref cannot verify this bundle |

If the engine declines the graph, the test skips in every mode, or fails when a support claim promised support. `--validator` chooses where the comparison runs (see [Verification modes](../../../../dnn-providers/integration-tests/docs/running-tests.md#verification-modes)). Checking golden data against a reference executor is not a mode: `hipdnn_golden_data_tests` does it.

**Floating-point edge case**: `-0.0` vs `+0.0` uses value comparison, not bitwise. NaN handling is covered in [Data Integrity](#data-integrity).

#### Tolerance Framework

Tolerance for engine output is determined by a **three-level priority chain**. The highest-priority source that provides a value wins. Levels 1 and 3 are built; level 2 is not yet built.

##### Three levels

| Priority | Source | Set by | When it's used |
|----------|--------|--------|----------------|
| 1 | TOML per-engine override | Engine developer | An engine needs to temporarily tighten or loosen tolerance without regenerating bundles (e.g., debugging a specific kernel, or a fused kernel that legitimately trades precision for throughput) |
| 2 | Bundle tolerance (`meta.json`) | Generation/calibration tool | Steady state -- each bundle carries its own calibrated tolerance, set at generation time by measuring expected variance |
| 3 | Per-operation default (code) | Compile-time constant | Safety net for bundles without calibrated tolerance (e.g., bundles captured from C++ graph tests) |

If a TOML override matches, it wins. Otherwise the bundle's `meta.json` tolerance applies. If the bundle has no tolerance metadata, the per-operation default applies. Without level 2, the harness takes the per-operation default from [`ToleranceResolver.hpp`](../../../../dnn-providers/integration-tests/src/harness/tolerance/ToleranceResolver.hpp) and then applies a matching TOML override.

Each engine's config (e.g., [`MIOPEN_ENGINE.toml`](../../../../dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml)) can declare `[[tolerance_overrides]]` entries with glob-pattern filters and atol/rtol values. A `[[validator_overrides]]` entry selects a different validator for one output tensor, matched by tensor name. TOML overrides apply only to engine output. `hipdnn_golden_data_tests` uses the per-operation default only, so an engine config can never loosen the check on the golden data.

##### Bundle tolerance (level 2)

Not yet built. No code reads a `tolerance` block from metadata, and no calibration tool exists.

A global tolerance model -- one number per operation type -- is unlikely to work well across all bundle configurations. It will either be conservative enough that some checks become too loose (hiding regressions), or tight enough that numerically valid cases become flaky. Instead, each bundle carries its own calibrated tolerance in `meta.json`, set at generation time:

```json
{
  "tolerance": {
    "absolute": 3.2e-6,
    "relative": 1e-4,
    "deterministic": true,
    "calibration": {
      "method": "empirical_p99.9",
      "num_runs": 100,
      "calibrated_on": "MI300X",
      "calibrated_at": "2026-05-20T14:30:00Z"
    }
  }
}
```

**Calibration procedure** (performed by the generation tool):

1. Generate the golden reference output using the appropriate [reference strategy](#generation-pipeline)
2. Run the target engine N times (e.g., N=100) on the same input
3. For each run, compute element-wise absolute and relative error against the reference
4. Set `tolerance.absolute` = p99.9 of absolute errors across all elements and runs
5. Set `tolerance.relative` = p99.9 of relative errors
6. Record calibration metadata for provenance

For **deterministic engines** (DUT variance = 0), all N runs produce identical output. The calibrated tolerance covers only the reference-to-DUT precision gap and is tight.

For **non-deterministic engines** (e.g., atomic FP32 accumulation with scheduling-dependent order), runs produce different outputs. The calibrated tolerance covers both the precision gap and the DUT's run-to-run variance. This is the primary mitigation for non-deterministic engine flakiness -- see [Risk Register](#risk-register).

##### Per-operation default (level 3)

The per-operation default is the safety net. It starts simple and gets smarter:

- **Fixed tolerances**: The harness reads the operation type of each node and the output data type from the graph and uses them as a lookup key into [`TestTolerances.hpp`](../../test_sdk/include/hipdnn_test_sdk/utilities/TestTolerances.hpp), which defines compile-time constants per operation and data type (e.g., batchnorm inference fp32 = `2e-4`, conv fwd fp32 = `1e-5`). For a graph with several nodes, it takes the loosest per-node value. An unknown operation gets `1e-3`. The same value is used for atol and rtol. The graph does not compute tolerances — it provides the key, the lookup table provides the value.

- **Future -- dynamic tolerances**: [`DynamicTolerances.hpp`](../../test_sdk/include/hipdnn_test_sdk/utilities/DynamicTolerances.hpp) will replace fixed lookup values with tolerances computed from graph properties (operation, data type, tensor dimensions). This is the intended long-term solution but requires validation before it can replace fixed values (see [Future Work](#future-work)).

This evolution changes *how* the default is computed -- it does not change the three-level chain.

**Acceptance criteria**:
- [ ] Bundle tests follow the three-level tolerance chain: TOML override → bundle metadata → per-operation default
- [x] Per-engine TOML overrides apply to bundle engine tests -- no separate override mechanism
- [ ] Generation tool writes calibrated tolerance to `meta.json` using the empirical calibration procedure
- [x] Bundles without tolerance metadata fall back to per-operation default (no crash, no skip)

#### Data Integrity

Internal consistency is guaranteed by construction for generated bundles: `Graph.save()` writes single-graph JSON and `.bin` files from the same in-memory graph in a single call, so UIDs and tensor data correspond. Template-sweep tooling must provide the same guarantee after expansion: each row in `sweep.json` expands to a valid graph, and tensor data is loaded from that case's `golden/{CaseId}/` directory or explicit `golden.path`, never from the template directory. Corruption can only enter after generation -- partial downloads, disk errors, manual edits, or incorrect sweep rows.

Three checks catch the real tensor-data failure modes:

1. **Tensor size validation (load time)** -- the harness reads each `.bin` file with `tensorFromFileAndAttributes()`, which throws when the byte count does not match `element_space x sizeof(element_type)` declared in the expanded graph JSON (see [binary tensor format](#binary-tensor-format)). The test fails. This is the primary load-time integrity check -- fast and sufficient to catch truncated downloads and wrong-file swaps.

2. **NaN/Inf rejection (generation time)** -- reject output tensors containing NaN or Inf before writing any files. `generate_sdpa_fwd_golden.py` does this; `generate_batchnorm_reference.py` does not. All-same-value tensors are valid (e.g., a bias-only layer can produce uniform output).

3. **NaN/Inf rejection (verifier)** -- safety net for bundles generated by external tools. The **bundle verifier** scans floating-point input and output tensors and rejects any containing NaN or Inf. Ragged outputs keep NaN in their pad rows ([RFC 0014](0014_RaggedTensors.md)), so only Inf is an error there.

The **bundle verifier** (`reference-data-scripts/verify_golden_bundles.py`) runs checks #1 and #3 across the bundle and sweep directories given on its command line. It is a CLI that authors run before commit (see [Check the data itself](../../../../dnn-providers/integration-tests/docs/adding-tests.md#check-the-data-itself)). A pre-commit hook is not yet built, and CI does not run the verifier on the bundle tree. `--require-data` makes a `.bin` file that a DVC pointer lists but that is not on disk an error instead of a warning. DVC content-addressing handles file integrity, so no separate checksum is needed. The verifier also validates structural conventions and data-size budgets:

4. **Tier folder validation** -- warn about top-level directories that are not one of the four tier names
5. **Single-graph JSON validation** -- verify each graph file (`<BundleName>/<BundleName>.json`) is a parseable graph, and warn about any other JSON file (for example a stray `README.json` or editor config)
6. **Template-sweep validation** -- verify `graph.template.json`, `sweep.json`, placeholder coverage, per-case tensor UID/dims/strides/data-type coverage, duplicate ids, golden DVC pointer paths, and expanded graph schema validity
7. **Missing `.bin` files** -- for single-graph bundles and template cases that have a DVC pointer, report every tensor UID in the graph that has no `.bin` file in that bundle or case directory
8. **Forward-backward consistency** -- not yet built. For backward bundles, verify that `meta.json` includes a `forward_source` field (see [Forward-Backward Generation Constraint](#forward-backward-generation-constraint))
9. **DVC payload budget** -- warn when one bundle directory, or the golden directory of one sweep case, holds more than 1 MiB, and fail above 2 MiB. Not yet built: report the aggregate tensor payload for the tree and fail when committed golden tensor payload exceeds the 800 MB project budget from [Bundle size considerations](#bundle-size-considerations).

**Acceptance criteria**:
- [x] All checks implemented with actionable error messages naming the file and tensor UID
- [x] File size mismatch and NaN/Inf in golden data are hard FAILs, not warnings
- [ ] Pre-commit bundle verifier validates full and graph-only bundles before commit
- [ ] Pre-commit and CI bundle verification report per-case and aggregate tensor payload and enforce the warning/fail thresholds from [Bundle size considerations](#bundle-size-considerations)
- [x] Stray non-graph `.json` files and unexpected top-level directories produce warnings

---

### 4.4 CLI and Configuration

`hipdnn_integration_tests` takes these bundle flags (see the [command-line reference](../../../../dnn-providers/integration-tests/docs/running-tests.md#command-line-reference--hipdnn_integration_tests)). `--verification-mode` controls **what** the engine output is compared with. `--golden-data-dir` sets the bundle root in every mode.

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--vm, --verification-mode` | `auto`, `golden`, `gpu`, `cpu` | `auto` | Which reference source to use (see [Verification Modes](#verification-modes)) |
| `--gd, --golden-data-dir` | path | `<exe_dir>/../lib/integration-test-bundles` | Where to find bundle data |
| `--validator` | `auto`, `cpu`, `gpu` | `auto` | Where the comparison runs |
| `--no-bundles` | flag | off | Register no bundle tests; run only the C++ tests built into the binary |
| `--capture-bundles` | path | none | Write each C++ graph test as a bundle (`{suite}/{case}/{case}.json` + `.meta.json`) |

`hipdnn_golden_data_tests` takes `--reference cpu|gpu|both` (default `both`), `--golden-data-dir`, and `--validator`.

Flags have environment variable fallbacks:

| Flag | Environment variable |
|------|---------------------|
| `--verification-mode` | `HIPDNN_TEST_VERIFICATION_MODE` |
| `--golden-data-dir` | `HIPDNN_TEST_GOLDEN_DATA_DIR` |
| `--validator` | `HIPDNN_TEST_VALIDATOR` |
| `--no-bundles` | `HIPDNN_TEST_ALLOW_BUNDLES=0` |

For `--verification-mode`, `--golden-data-dir`, and `--validator`, the CLI flag takes precedence when both are set. `HIPDNN_TEST_ALLOW_BUNDLES` overrides `--no-bundles`.

**Acceptance criteria**:
- [x] Flags parsed and stored in `TestConfig` singleton
- [x] Environment variable fallbacks work when CLI flag is absent
- [x] `--verification-mode golden` fails tests that have no golden data
- [x] `--verification-mode gpu` and `--verification-mode cpu` do not compare against golden data

---

## Integration

### CI Integration

[`.github/workflows/hipdnn-superbuild-ci.yml`](../../../../.github/workflows/hipdnn-superbuild-ci.yml) runs `dvc pull --recursive dnn-providers/integration-tests/integration-test-bundles` in its Windows and Linux lanes, before CMake configure. The step fails when the pull produces no `.bin` file, so a lane cannot pass with no golden data. The lanes then run `ctest --test-dir build --output-on-failure`, which runs every registered test, including `hipdnn_golden_data_tests` and the provider lanes of `hipdnn_integration_tests`. No workflow sets `--verification-mode`, so CI uses `auto`.

Tiers are determined by the top-level folder (see [Folder Convention](#folder-convention)), following the same [tier cascade](../../../../dnn-providers/integration-tests/docs/running-tests.md#how-tiers-cascade) as all other integration tests. [`test_categories.yaml`](../../../../dnn-providers/integration-tests/test_categories.yaml) sets the tier labels for `hipdnn_golden_data_tests`. Each provider's `test_categories_integration.yaml` sets them for its lane.

#### Planned CI stages

Not yet built. Each stage pulls only the bundle data for its tiers (`dvc pull integration-test-bundles/{tier}/`) and runs one tier label:

| CI Stage | ctest Command | Verification Mode | Notes |
|----------|--------------|-------------------|-------|
| Pre-submit (smoke) | `ctest -L quick` | `auto` | `quick/` tier tests; tests without golden data fall back to GPU/CPU ref |
| Post-submit | `ctest -L standard` | `auto` | `quick/` + `standard/` tier tests; fallback to computed for gaps |
| Nightly | `ctest -L comprehensive` | `golden` | All tiers up to comprehensive; `golden` mode fails tests without golden data |
| Weekly | `ctest -L full` | `golden` | All tiers |

### Workflows

**Add a test -- any operation, any data type** (developer):
1. Get a graph: capture it from a C++ graph test with `--capture-bundles`, or write a generation script following the [`generate_batchnorm_reference.py`](../../../../dnn-providers/integration-tests/reference-data-scripts/generate_batchnorm_reference.py) pattern to produce golden data
2. Place it with `migration-scripts/import_graph.py`, which writes the graph, the metadata, and an empty `support.json`, and prints the GTest name (see [Add a bundle](../../../../dnn-providers/integration-tests/docs/adding-tests.md#add-a-bundle))
3. Run `verify_golden_bundles.py` on the changed directories to verify the bundle is well-formed
4. Re-run CMake and run the tests. No C++ changes, no recompile -- the generic runner discovers the bundle automatically. Commit the JSON and DVC pointer files to git and push tensor data to DVC

**Debug a customer issue** (support):
1. Receive the customer's bundle folder -- no source code or NDA required
2. Drop the folder into any tier directory (e.g., `integration-test-bundles/quick/customer_issues/CustomerBundle/`) and re-run CMake. A folder with golden data needs a `.meta.json`
3. Run tests -- the runner picks it up, executes the engine, compares against golden output
4. Inspect the diff: which tensors diverge, by how much, at which indices

**Reproduce a CI failure locally** (developer):
1. Pull the golden data via DVC, then re-run CMake
2. Run the failing test with `--gtest_filter=<suite>.<test>` from the CI log
3. The bundle is self-contained -- no environment-specific setup beyond the engine under test

**Add golden test support for a new plugin** (provider developer):
1. Call `add_external_integration_test_target()` from `dnn-providers/integration-tests/cmake/HipdnnIntegrationTestHelpers.cmake` in your provider's CMake (see [Add an engine lane](../../../../dnn-providers/integration-tests/docs/adding-tests.md#add-an-engine-lane))
2. The same bundles are automatically validated against your plugin -- no new test data needed

---

## Migration Plan

The generic runner replaces per-operation test classes in phases. Each phase ships on its own -- no flag day. C++ graph tests (`IntegrationGraphVerificationHarness`) stay for special cases and build only with `-DBUILD_CPP_GRAPH_TESTS=ON`.

| Phase | What ships | Status |
|-------|-----------|--------|
| 1 | Generic runner, discovery, tier folders. Batchnorm bundles in `integration-test-bundles/quick/`. | Built |
| 2 | Three-level tolerance chain (TOML override -> bundle metadata -> per-operation default). Bundle verifier. | Partly built: TOML override, per-operation default, and the verifier CLI. Bundle-metadata tolerance and the pre-commit hook are not yet built. |
| 3 | Bundles for conv, matmul, and pointwise. Generic runner covers these operations end-to-end. | Built |
| 4 | CI integration: DVC pull, per-tier minimum test count baseline, `--verification-mode auto` as default. | Partly built: DVC pull and `auto`. A per-tier baseline is not yet built; a run that selects zero tests fails. |
| 5 | Remaining operations (SDPA, layernorm, RMS norm, reduction). SDPA onboarding requires the matched-precision reference strategy, the forward-backward generation constraint, empirically calibrated tolerance for non-deterministic engine variants, and the tolerance calibration tool — see the relevant sections in Detailed Design. | Partly built: bundles exist for these operations. The forward-backward constraint and tolerance calibration are not yet built. |

**Rollback**: At any phase, setting `--verification-mode cpu` (or `gpu`) stops engine tests from using golden data.

---

## Data Management

Golden data lives in two places -- **source tree** and **runtime**:

- **Source tree**: Bundle data lives at `dnn-providers/integration-tests/integration-test-bundles/`.

- **Runtime**: CMake copies the bundles to `<build>/lib/integration-test-bundles/` at configure time and installs them. The default path is `<exe_dir>/../lib/integration-test-bundles/`. The `--golden-data-dir` CLI flag or `HIPDNN_TEST_GOLDEN_DATA_DIR` env var overrides this location.

##### Hybrid storage: git for metadata, DVC for tensors

Graph files (`.json`), template files (`graph.template.json`), sweep files (`sweep.json`), metadata (`meta.json`), and DVC pointer files (`*.dvc`) are checked into **git** -- they are small, human-readable, and benefit from standard diff/review tooling. Binary tensor files (`.bin`) are stored in **DVC** (Data Version Control), which the repo already uses for large binary assets. This split keeps the graph definition, sweep case data, and provenance versionable in git while offloading large opaque binaries to content-addressed storage.

##### Compression scope

Template-sweep bundles compress graph metadata in git, not tensor data in DVC. For the same cases, the raw `.bin` tensors, DVC object count, S3 footprint, and CI transfer cost are unchanged; only repeated graph JSON files and near-identical bundle directories disappear. For example, an SDPA forward set may collapse from ~36 bundle directories to ~4 templates, while DVC storage remains the same because each case still owns its golden tensors.

##### Bundle size considerations

Bundle sizes vary by orders of magnitude across operations:

| Operation | Typical bundle size | Tensors |
|-----------|-------------------|---------|
| Batchnorm forward | < 1 MB | 6 |
| SDPA forward (medium shape) | ~32 MB | 5 |
| SDPA backward (medium shape) | ~32 MB | 9 |
| SDPA backward (large shape) | 1+ GB | 9 |

Golden tensor payload under `dnn-providers/integration-tests/integration-test-bundles/` should remain below 1 GB total. The bundle verifier warns when one bundle or sweep case holds more than 1 MiB and fails above 2 MiB. A check that fails when the aggregate committed tensor payload exceeds 800 MB is not yet built. Cases larger than the per-case warning need an explicit rationale in review. When tensors are reproducible from `(shape, dtype, layout, mask/features, seed)` and static data is not needed to reproduce a bug, prefer a graph-only bundle with runtime generation over committing large `.bin` payloads.

##### Shape selection policy

Generators should produce a **representative subset** of shapes per operation, not an exhaustive matrix. The goal is to cover tolerance-critical corners:

- **Small shapes**: where relative error dominates (e.g., single-element edge cases)
- **Medium shapes**: typical production workloads
- **Large shapes**: where accumulation length stresses the math (e.g., long sequence SDPA)

Not every combination of batch × heads × sequence × dim is needed. Select shapes that exercise distinct numerical regimes.

##### Tier placement

Tier placement is governed by **runtime budget**, not bundle size, following [TheRock's test filtering model](https://github.com/ROCm/TheRock/blob/main/docs/development/test_filtering.md#types-of-filters). A small bundle with an expensive engine may belong in `standard/`; a large bundle with a fast engine may fit in `quick/`. An auto-tier classifier that times test execution and suggests placement is not yet built (see [Future Work](#future-work)).

##### DVC pull at CI time

DVC is already in the repo for other large binary assets, content-addressing provides integrity guarantees, and selective fetch by path avoids pulling data that a run does not need. CI pulls the whole bundle tree before configure (see [CI Integration](#ci-integration)). Developers can fetch specific operations: `dvc pull -R dnn-providers/integration-tests/integration-test-bundles/quick/SdpaFwd`. Without golden data, `auto`, `cpu`, and `gpu` modes still check every bundle against a reference executor. CI does not fall back: a pull that fetches nothing fails the lane, because a green run that checked no golden data is the worse failure.

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| FlatBuffers schema change | Old JSON bundles unreadable by `loadGraphAndTensors()` | Low | Schema changes must be backwards compatible -- old graphs continue to work when new fields are added. FlatBuffers supports this natively via optional fields |
| Reference script bug freezes wrong data | Silent incorrect baseline | Medium | Cross-validate against C++ CPU ref (`hipdnn_golden_data_tests`); review generated data before committing; [generation-time and verifier NaN/Inf checks](#data-integrity) reject degenerate outputs |
| PyTorch version drift | Different versions produce slightly different outputs | Low | Pin PyTorch version in `requirements.txt`; regenerate when upgrading |
| Large golden data sets slow CI | CI feedback loop degrades | Low | Storage caching, selective fetch by path, the per-case DVC payload budget check (the aggregate check is not yet built), and graph-only/runtime generation for reproducible large cases |
| Remote storage unavailable | Golden-data CI lanes fail | Low | The CI fetch step fails the lane on purpose rather than pass with no golden data. Locally, `auto`, `cpu`, and `gpu` modes run without golden data |
| Silent coverage regression | Accidental bundle deletion reduces test count; CI stays green | Low | Code review; DVC history tracks all bundle changes; CI fails when `dvc pull` produces no `.bin` file; `hipdnn_integration_tests` exits 1 when zero tests run |
| Non-deterministic engine flakiness | Golden comparison fails intermittently for engines using non-deterministic accumulation (e.g., atomic FP32 additions where thread-block scheduling varies run-to-run) | Medium | Tolerance must cover DUT run-to-run variance, calibrated empirically from repeated runs -- not hand-tuned. The planned `deterministic` metadata field aids triage but is not the mitigation. CI must exercise the production engine path, not a deterministic substitute |
| Reference-side non-determinism | If GPU-based references are added to the golden data pipeline, non-deterministic accumulation could contaminate the golden snapshot itself -- making it one sample from a distribution rather than a stable value | Low (not a concern today -- references are CPU/PyTorch-based) | Golden generation must use deterministic paths. If GPU references are ever needed, constrain them to deterministic kernel variants or use statistical tolerance banding from N runs |

---

## Known Limitations

Comparison testing can confirm that two implementations agree, not that either is correct. If the reference executor and the engine under test share the same bug, the test passes.

---

## Alternatives Considered

The following architectural forks were considered during design. Each records the rationale for the chosen path.

- **Per-tensor `.bin` files vs. single archive (npz / HDF5 / zip).** A single archive per bundle would simplify "share = one file" and reduce filesystem entry counts, but trades off transparent diff/inspection and tool-agnostic readability. *Why we chose per-tensor `.bin`: individual files are inspectable with standard tools (hexdump, NumPy `fromfile`), diffable in reviews, and independently fetchable by DVC. The "share one folder" workflow is simple enough.*

- **Raw `.bin` files vs. self-describing tensor formats (safetensors, NumPy `.npy`).** Self-describing formats embed dtype, shape, and strides in a header, making each file independently interpretable. Our `.bin` files are raw data — all metadata lives in the graph JSON. *Why we chose raw `.bin`: the graph JSON already carries dtype, dims, and strides for every tensor. A self-describing format would duplicate that metadata and add an external dependency without clear benefit. The existing `LoadGraphAndTensors` loader already handles this format.*

- **JSON graph + binary tensors vs. single FlatBuffers binary that embeds tensor data.** A unified binary would eliminate the JSON<->binary integrity surface, but costs human readability of the graph definition. *Why we chose split format: the graph JSON is human-readable and diffable in PRs, which matters for review and debugging. Binary tensor data changes frequently; the graph definition changes rarely. Splitting them lets DVC track the large binaries while the graph stays in git.*

- **Recursive auto-discovery vs. explicit index manifest.** A manifest (e.g., `bundles.toml` listing every bundle) makes test sets explicit and detects accidental drops/removes, but adds a step to the "drop a folder, done" workflow. *Why we chose auto-discovery: "drop files, run tests" is the primary workflow. A manifest adds a maintenance step and a failure mode (manifest out of sync with disk). Accidental deletions are caught by code review and DVC history.*

- **Graph-derived names vs. path-derived names vs. explicit case ids.** Path-derived names are stable against graph edits and need no file parsing at discovery. Pure graph-derived names are stable against folder reorganization but do not distinguish many same-topology sweep rows. *Why we chose path-derived names plus explicit case ids: discovery names a test without opening the file, so `--gtest_filter` drops excluded cases before they are loaded, and any folder (for example a customer drop) gets a usable name. Single-graph bundles use the folder path as the suite and the file stem as the test; template sweeps use the sweep folder as the suite and the explicit `cases[].id` as the test. Collisions are detected at discovery time with a clear error.*

- **DVC vs. Git LFS vs. pre-staged CI artifact.** DVC is already in the repo for other large assets; LFS is more universal but adds a new dependency; a pre-staged artifact decouples test latency from data fetch but loses content-addressing. *Why we chose DVC: already in the repo for other large assets, content-addressing provides integrity guarantees, and selective fetch by path avoids pulling data for operations not under test.*

- **Per-test checksum verification vs. DVC content-addressing.** Per-tensor SHA-256 checksums stored in JSON would catch corruption at any stage, but DVC already content-addresses every tracked file. *Why we chose DVC: already in the repo, no extra metadata to maintain, and tensor size validation at load time catches the practical failure modes (truncation, wrong file).*

- **Tier as top-level folder vs. tier as bundle metadata.** Folder-as-tier maps cleanly to `ctest -L`, but moving a bundle between tiers requires `git mv`. Metadata-as-tier keeps location stable but requires reading every JSON to build the tier list. *Why we chose folder-as-tier: maps directly to `ctest -L` without reading any JSON, matches how the rest of the integration suite assigns tiers, and makes tier membership visible in the directory listing.*

---

## Future Work

1. **Dynamic tolerance integration**: Wire the existing `DynamicTolerances` functions (matmul, conv, batchnorm, layernorm, RMS norm, pointwise, SDPA) into the bundle harnesses as the per-operation default.
2. **Engine-as-bundle-producer**: Any engine can take a graph-only bundle, execute the graph, and write the results back as a full bundle -- enabling cross-engine comparison without a shared reference source.
3. **Bundle-to-bundle comparison**: A standalone tool that loads two bundles and diffs their output tensors directly, matched by graph content rather than filename.
4. **Auto-tier classification**: The generator suggests the appropriate tier folder based on tensor element counts, matching the existing size conventions.
5. **KnobSettings coverage**: Validate the same golden bundles under different engine KnobSettings configurations to catch regressions from tuning changes.
6. **Tolerance calibration tool**: Extend the generation framework to run the target engine N times on each bundle, compute empirical p99.9 tolerance bounds, and write calibrated tolerance values to `meta.json`. Required before onboarding operations with non-deterministic engines (e.g., SDPA backward with atomic accumulation).
7. **Structural template-sweep composition**: If several template sweeps become near-duplicates because their graphs differ only by optional tensors, node inputs, or node outputs (for example SDPA stats, FP8 descale tensors, or GROUP sequence-length tensors), extend `graph.template.json` and `sweep.json` with an explicit structural templating mechanism so those sweeps can be combined. This is intentionally out of v1 until real duplication justifies the extra schema and validation complexity.
