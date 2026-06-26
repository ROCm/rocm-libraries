# RFC 0011: Golden Reference Validation

> Owner: Integration Test Team
> Last updated: 2026-06-22

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
2. **Coverage gap**: Operations not yet implemented in the reference executor cannot be tested (e.g., SDPA has no C++ reference kernel)
3. **Non-determinism**: GPU reference results can vary across runs, making failure investigation harder
4. **Slowness**: CPU reference execution for large tensors is the bottleneck in full-tier tests

A prior effort established a golden reference pattern -- golden data bundles (graph JSON + tensor `.bin` files) loaded from disk and validated against engine outputs. The initial infrastructure is in place for batchnorm. This RFC extends golden data coverage to all operation types, formalizes the folder convention, adds data integrity checks, and integrates with CI.

**Long-term direction**: Every integration test becomes a data bundle. Adding a test means dropping files in a folder -- no C++ code, no per-operation test classes. The regression suite's end state is entirely data-driven.

---

## Design Overview

Golden reference validation has two pipelines -- [**generation**](#generation-pipeline) (produces bundles) and [**validation**](#generic-test-runner) (consumes them) -- sharing one logical data model: a graph plus optional golden tensor data. The on-disk format supports two bundle kinds. **Single-graph bundles** keep the original `{Name}.json` + `{Name}.tensor{uid}.bin` layout for one-off/customer cases. **Template-sweep bundles** store one `graph.template.json` plus a `sweep.json` case matrix for many cases that share one graph topology.

**Design principle -- test identity is explicit and stable.** Single-graph bundles use the legacy bundle directory name as the gtest test name. Template-sweep bundles derive the suite from the sweep location and use the sanitized `cases[].id` as the test name. Folder hierarchy is used for discovery and grouping, but compressed bundles do not depend on a unique graph file path per case.

Compressed template sweeps are **git-side compression only**. They collapse repeated graph JSON files and directory clutter into one template plus a sweep table; they do not reduce tensor `.bin` payload, DVC/S3 storage, or CI transfer volume for the same set of golden tensors.

---

## Existing Infrastructure

The golden reference pattern is already in production for batchnorm: six bundles spanning four layout/datatype combinations are validated on every CI run by per-operation C++ test classes that route them through either the CPU reference executor or the MIOpen plugin.

### What exists today

- **A bundle format and a C++ loader.** The on-disk shape -- graph JSON plus one `.bin` per tensor -- is defined, and `loadGraphAndTensors()` reads bundles from disk and separates inputs from expected outputs.
- **Two test fixtures.** One runs a bundle through the in-tree reference executor; the other runs it through the GPU plugin. Each operation that wants golden coverage today writes its own C++ test class that wires specific bundles into GTest.
- **Tolerance machinery, three layers, partially wired.**
  - *Per-operation compile-time defaults* (atol/rtol constants per operation + datatype).
  - *Per-engine TOML overrides* with glob matching, used today by the broader integration suite but not yet by golden tests.
  - *Dynamic Higham-style error bounds* -- functions exist for the major operations but aren't wired into anything yet.
- **A Python generation framework** that produces bundles using PyTorch as the reference source.

### What this RFC adds on top

1. **A generic runner** -- one base class plus three execution-step subclasses (CPU ref, GPU ref, engine) -- that replaces the per-operation test classes. Bundles are discovered recursively at GTest startup; adding an operation no longer requires new C++.
2. **A tolerance lookup chain** that connects golden tests to the existing TOML override mechanism, falling back to the per-operation default. Dynamic tolerances stay future work.
3. **Integrity checks** at generation, load, and pre-commit time. See [Data Integrity](#data-integrity).
4. **A formalized folder convention** with the same tier cascade as the rest of the integration suite. See [Folder Convention](#folder-convention).
5. **Relocation** of the existing batchnorm bundles from `projects/hipdnn/hipdnn_reference_data/` to `dnn-providers/integration-tests/integration_test_bundles/quick/BatchnormFwdInference/`, so all integration test data lives together.
6. **Compressed template sweeps** for shared-topology bundles, so dtype/layout/shape coverage can be added as rows in `sweep.json` instead of as repeated graph JSON files. This reduces review noise and filesystem clutter, not DVC tensor-data size.

### Code map for implementers

The functional changes are described in the relevant Detailed Design subsections; this table is just a pointer to the affected files.

| File | Disposition | Notes |
|------|-------------|-------|
| [`LoadGraphAndTensors.hpp`](../../test_sdk/include/hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp) | Kept, extended | Add tensor-size check after load ([Data Integrity #1](#data-integrity)) |
| [`GoldenReferenceCpu.hpp`](../../test_sdk/tests/utilities/GoldenReferenceCpu.hpp) | -> `TestCpuReferenceUsingGoldenValues.hpp` | Subclass; CPU reference executor execution step |
| (new) `TestGpuReferenceUsingGoldenValues.hpp` | New | Subclass; GPU reference executor execution step |
| [`GoldenReferenceGpu.hpp`](../../../../dnn-providers/miopen-provider/tests/common/GoldenReferenceGpu.hpp) | -> `IntegrationGpuGoldenReferenceEngineValidation.hpp` | Subclass; engine execution step, "unsupported" -> SKIP |
| [`TestCpuFpReferenceBatchnorm.cpp`](../../test_sdk/tests/utilities/TestCpuFpReferenceBatchnorm.cpp) | Removed | Per-operation C++ class no longer needed |
| [`TestTolerances.hpp`](../../test_sdk/include/hipdnn_test_sdk/utilities/TestTolerances.hpp) | Kept, extended | Generic runner adds operation-type lookup |
| [`TestSettings.hpp`](../../../../dnn-providers/integration-tests/src/harness/TestSettings.hpp), engine `.toml` files | Kept as-is | Existing TOML overrides apply to golden tests unchanged |
| [`DynamicTolerances.hpp`](../../test_sdk/include/hipdnn_test_sdk/utilities/DynamicTolerances.hpp) | Future | Wired in later as the per-operation default |
| [`reference_data_scripts/utilities/`](../../reference_data_scripts/utilities/) | Kept, enhanced | Generators auto-derive output path from graph content |
| [`hipdnn_reference_data/`](../../hipdnn_reference_data/BatchnormFwdInference/) (6 batchnorm bundles) | Relocated | Moves to `integration_test_bundles/quick/BatchnormFwdInference/` |
| (new) `IntegrationGraphGoldenReferenceVerificationHarness` | New | Owns discovery, loading, tolerance lookup, comparison |

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

Provenance and calibration data is stored in a **separate companion file** (`{Name}.meta.json`), not in the graph JSON. This keeps the graph schema 1:1 with the hipDNN `graph.fbs` format and avoids polluting the computation definition with test infrastructure details. The test runner loads this file to read calibrated tolerance values (see [Tolerance Framework](#tolerance-framework)); all other fields are for humans, tooling, and traceability.

```
{Name}/
  {Name}.json              # Graph definition (unchanged)
  {Name}.tensor{uid}.bin   # Tensor data (unchanged)
  {Name}.meta.json         # Provenance + calibrated tolerance (loaded by runner for tolerance)
```

**Portable bundle** (CPU/framework reference — runs on any ASIC):

```json
{
  "format_version": 1,
  "generator": "reference_data_scripts/batchnorm_inference.py",
  "generator_version": "1.0.0",
  "generated_at": "2026-05-11T14:30:00Z",
  "rocm_version": "6.4.0",
  "reference_source": "PyTorch 2.3.0",
  "reference_source_hash": "a3f8c2e1",
  "reference_strategy": "precision_uplift",
  "generation_command": "python generate_batchnorm_reference.py --name typical",
  "notes": "baseline for RFC 0011 migration",
  "seed": 42,
  "minimum_vram_mb": 8192
}
```

**Arch-specific bundle** (GPU reference — only valid on the generating ASIC):

```json
{
  "format_version": 1,
  "generator": "reference_data_scripts/generate_sdpa_fwd_golden.py",
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

The `.meta.json` file itself is optional for forward bundles. A bundle without one is valid: it falls back to the per-operation default tolerance (see [Tolerance Framework](#tolerance-framework)) and is never skipped or rejected on that basis — this is the expected state for bundles migrated from the old test system. When a `.meta.json` *is* present, the `generator`, `reference_source`, and `reference_strategy` fields are mandatory and the pre-commit bundle verifier rejects a `.meta.json` missing them. Remaining fields are optional. Generator scripts populate the metadata file automatically. (Backward bundles are the exception: they must include a `.meta.json` with a `forward_source` field — see [Forward-Backward Generation Constraint](#forward-backward-generation-constraint).)

**Field reference:**

| Field | Required | Type | Purpose |
|-------|----------|------|---------|
| `format_version` | Yes | integer | Schema version. Must be `1`. Reader rejects files with missing or unsupported versions to allow future schema evolution. |
| `generator` | Yes | string | Path to the script that produced this bundle. |
| `generator_version` | No | string | Version of the generator script. |
| `generated_at` | No | string | ISO 8601 timestamp of generation. |
| `gpu_architecture` | No | string | GPU arch that produced the reference data (e.g. `"gfx942"`). When present, the arch guard skips the test if the current device doesn't match. When absent, the data is assumed portable across all ASICs. Only set this when the reference source is a GPU-specific tool whose numerical output varies by architecture. |
| `rocm_version` | No | string | ROCm version used during generation. |
| `reference_source` | Yes | string | What computed the reference output (e.g. `"PyTorch 2.3.0"`, `"AITER 0.1.13"`, `"CpuReferenceGraphExecutor"`). Informational — used for traceability, not for guard logic. |
| `reference_source_hash` | No | string | Commit hash or checksum of the reference source for traceability. |
| `reference_strategy` | Yes | string | How reference precision was chosen (see table below). |
| `generation_command` | No | string | Full command line used to generate the bundle. |
| `notes` | No | string | Free-text notes for humans. |
| `seed` | No | integer | RNG seed used for input generation. Enables reproducible regeneration. |
| `minimum_vram_mb` | No | integer | Minimum device VRAM in MB required to run this bundle. The runner skips the test if the device has less. |

Unknown fields are ignored by the reader, so generators may include additional fields without breaking compatibility.

`reference_strategy` records how the reference precision was chosen (see [Generation Pipeline](#generation-pipeline)):

| Value | When to use |
|-------|-------------|
| `precision_uplift` | Shallow operations (batchnorm, pointwise). Reference computed in higher precision than target, cast down. |
| `matched_precision` | Deep operations (SDPA backward). Reference computed with the same intermediate precision as the DUT's accumulation contract. |

#### Golden Data Format

A bundle is a **directory** containing files with a shared base name: one `.json` file (graph definition conforming to the [`graph.fbs`](../../flatbuffers_sdk/schemas/graph.fbs) schema) and one `.tensor{uid}.bin` file per tensor (raw contiguous data matching the tensor's declared `data_type`, `dims`, and `strides`). The existing `LoadGraphAndTensors.hpp` (C++ reader) and [`Graph.save()`](../../reference_data_scripts/utilities/graph.py) (Python writer) already implement this format.

```
{Name}/                    # One directory per bundle
  {Name}.json              # Graph definition (operation, tensor metadata, parameters)
  {Name}.tensor{uid}.bin   # Raw tensor data, one file per UID
```

For example, batchnorm inference bundles:

```
BatchnormFwdInference/nchw/fp32/
  typical/
    typical.json               # Graph: batchnorm inference, 6 tensors, fp32
    typical.tensor0.bin         # x (input)
    typical.tensor1.bin         # mean (input)
    typical.tensor2.bin         # inv_variance (input)
    typical.tensor3.bin         # scale (input)
    typical.tensor4.bin         # bias (input)
    typical.tensor5.bin         # y (output -- golden reference)
  large_batch/
    large_batch.json           # Same operation, large batch dimension
    large_batch.tensor0.bin
    ...
  resnet50_layer3/
    resnet50_layer3.json       # Same operation, shape from a real model
    resnet50_layer3.tensor0.bin
    ...
```

The name describes *why the test exists* -- `typical` covers a common shape, `large_batch` stresses the batch dimension, `resnet50_layer3` tests a real-world shape. Tensor dimensions are in the graph JSON, not the name. See [Folder Convention](#folder-convention) below for the full directory structure.

##### Binary tensor format

Each `.tensor{uid}.bin` file is a **raw dump of the tensor's underlying storage** -- no header, no metadata, no framing. The graph JSON carries all the metadata needed to interpret the bytes:

| Property | Source | Details |
|----------|--------|---------|
| Element type | `data_type` field in JSON | `FLOAT` = 4 bytes, `HALF` = 2 bytes, `BFLOAT16` = 2 bytes, etc. |
| Dimensions | `dims` field in JSON | Element counts per dimension (e.g., `[1, 3, 224, 224]`) |
| Strides | `strides` field in JSON | **Element strides**, not byte strides (e.g., `[150528, 50176, 224, 1]`). Multiply by `sizeof(element_type)` for byte offsets. |
| Byte order | Native platform | Little-endian on x86-64. No endianness marker in the file. |

The file size in bytes equals `element_space x sizeof(element_type)`, where `element_space` is computed from `dims` and `strides` (the total storage footprint including any gaps for non-contiguous layouts). For a contiguous (packed) tensor, `element_space` equals the product of `dims`.

**To read a `.bin` file**: allocate `element_space x sizeof(T)` bytes, read the file into that buffer, then index using the strides from JSON. For contiguous tensors this is a straightforward row-major (C-order) array.

**v1 scope: dense (packed) tensors only.** The binary format supports strided layouts but not sparse representations (CSR, COO, etc.). Sparse tensor support is deferred to a future revision.

#### Compressed Template Sweeps

Use a **template-sweep bundle** when several cases share the same graph topology and differ only by case data such as tensor dtype, dims, strides, layout fields, scalar attributes, metadata, or golden data. The template is required; one-off graphs and customer drops should stay in the single-graph format.

```
integration_test_bundles/{Tier}/{Operation}/{TopologyName}/
  graph.template.json
  sweep.json
  golden/{CaseId}/tensors.dvc
```

`TopologyName` is a human-readable name for one invariant graph skeleton. It is not a dtype, layout, shape, or case name. Good examples are `Inference`, `BhsdNoMask`, and `BhsdNoMaskStats`; poor examples are `fp16`, `nchw`, `4d`, or `small_fp32_nchw`.

`graph.template.json` is the invariant graph JSON topology with placeholders for per-case values. Placeholder syntax is `${case.<field>}`. Scalar placeholders resolve from `cases[].values`, tensor placeholders resolve from `cases[].values.tensors` by tensor UID, and node-attribute placeholders resolve from `cases[].values.attributes` as `${case.attributes.<field>}`. Attribute placeholders are allowed for values such as SDPA mask bounds and diagonal alignment.

The same topology means node count, edge wiring, tensor UID identity, tensor set, and operation sequence stay fixed. Rank changes are allowed in the same topology when they are represented only by per-case `dims` and `strides`. Changing nodes, edges, tensor UID identity, tensor count, or operation sequence creates a new `TopologyName`. v1 intentionally does not support per-case tensor opt-in/opt-out; SDPA stats output, FP8 descale tensors, and GROUP mode sequence-length tensors use separate topologies. If future graphs show enough duplication across those adjacent topologies to make this split painful, the template and sweep schema can be extended with structural conditionals or sweep composition so those cases can be combined deliberately instead of adding that complexity up front.

`sweep.json` owns the case matrix:

```json
{
  "version": 1,
  "cases": [
    {
      "id": "small_fp32_nchw",
      "values": {
        "layout": "nchw",
        "tensors": [
          {
            "uid": 0,
            "data_type": "FLOAT",
            "dims": [2, 3, 4, 5],
            "strides": [60, 20, 5, 1]
          },
          {
            "uid": 1,
            "data_type": "FLOAT",
            "dims": [1, 3, 1, 1],
            "strides": [3, 1, 1, 1]
          }
        ],
        "attributes": {
          "epsilon": 1e-5
        }
      },
      "golden": {
        "id": "small_fp32_nchw",
        "path": "golden/small_fp32_nchw/tensors.dvc"
      },
      "metadata": {
        "format_version": 1,
        "operation": "BatchnormFwdInference"
      }
    }
  ]
}
```

`cases[].id` is stable and unique within the sweep. It is the tooling handle, golden lookup key, failure-output case id, and gtest test name after sanitization. All cases from one sweep register in the same suite and differ by id only.

Case ids must be lowercase_snake_case and include the discriminator tokens developers need for filtering. At minimum, include a shape or scenario token plus dtype and layout when either varies in the sweep; include feature tokens such as `causal`, `stats`, `fp8`, or `group` when those features vary within the topology. Do not rely on directory names to make dtype/layout filters work for template sweeps.

Golden tensor data is stored under a per-case directory: `golden/{CaseId}/tensors.dvc`. When DVC pulls the tensor blobs, UID-based tensor filenames remain inside that case directory and cannot collide with another case that uses the same tensor UIDs. Sweep-aware loading must key tensor lookup on the logical `CaseId` or `golden.path` and read from `golden/{CaseId}/`; it must not derive `.bin` paths from the `graph.template.json` sibling directory.

Validation rules for template sweeps:

- duplicate `cases[].id` values are errors;
- case ids missing required discriminator tokens for varied dtype, layout, or feature fields are authoring errors;
- missing placeholder values are errors;
- unused `values` entries are warnings so authors can keep descriptive fields;
- duplicate tensor `uid` values within one case are errors;
- tensor `uid` values not present in the template graph are errors;
- cases that omit a template tensor UID, add a new tensor UID, or otherwise change the template tensor set are errors;
- template tensor placeholders for `dims`, `strides`, or `data_type` without matching case tensor entries are errors;
- template node-attribute placeholders without matching `cases[].values.attributes` entries are errors;
- missing `golden.path` for a non-null `golden` is an error;
- every expanded case must build and validate the same flatbuffer graph as a single-graph bundle before it can be registered.

#### Folder Convention

The top-level directory is organized by **tier**. Below that, the runner discovers both single-graph bundles and template-sweep bundles. Single-graph bundles keep the legacy `{Operation}/{Layout}/{DataType}/{BundleName}/` convention. Template sweeps use `{Operation}/{TopologyName}/` because layout, data type, dims, and strides are case data in `sweep.json`.

```
integration_test_bundles/
  {Tier}/                           # required: quick, standard, comprehensive, full
    {Operation}/{Layout}/{DataType}/{BundleName}/
      {BundleName}.json
      {BundleName}.meta.json
      {BundleName}.tensors.dvc
    {Operation}/{TopologyName}/
      graph.template.json
      sweep.json
      golden/{CaseId}/tensors.dvc
```

In the source tree, bundle data lives with the integration test suite at `dnn-providers/integration-tests/integration_test_bundles/`. At runtime, the test binary reads directly from the source tree location -- no CMake copy step. The default path is resolved relative to the executable, and `--golden-data-dir` or `HIPDNN_TEST_GOLDEN_DATA_DIR` can override it.

##### Tier folders

The top-level folder determines the tier. The runner scans each tier directory separately (see [Generic Test Runner](#test-discovery)), mapping to the standard [tier cascade](../../../../dnn-providers/integration-tests/README.md#test-tiers):

| Folder | GTest prefix | `ctest -L` |
|--------|-------------|------------|
| `quick/` | (none) | `quick` |
| `standard/` | `Standard` | `standard` |
| `comprehensive/` | `Comprehensive` | `comprehensive` |
| `full/` | `Full` | `full` |

Note: the ctest label uses `quick` for the smoke tier.

##### Recommended sub-structure

For single-graph bundles:

| Level | Convention | Examples |
|-------|-----------|----------|
| Operation | PascalCase, direction suffix | `BatchnormFwdInference`, `ConvFwd`, `ConvBwd`, `MatmulFwd` |
| Layout | Lowercase | `nchw`, `nhwc`, `ncdhw`, `ndhwc` |
| DataType | Lowercase abbreviation | `fp32`, `fp16`, `bfp16` |
| BundleName | lowercase_snake_case -- one directory per one-off bundle. Name describes why the test exists, not only tensor shapes. | `typical/`, `odd_spatial/`, `single_element/`, `resnet50_layer3/` |

For template-sweep bundles:

| Level | Convention | Examples |
|-------|-----------|----------|
| Operation | PascalCase, direction suffix | `BatchnormFwdInference`, `SdpaFwd` |
| TopologyName | PascalCase or mixed acronym style; names the invariant graph skeleton, not dtype/layout/shape. | `Inference/`, `BhsdNoMask/`, `BhsdNoMaskStats/` |
| CaseId | lowercase_snake_case in `sweep.json`; becomes the gtest test name after sanitization. Include scenario/shape plus dtype and layout when varied so filters remain stable. | `small_fp32_nchw`, `small_fp16_nchw`, `gqa_bf16`, `small_fp16_bottom_right_causal` |

This convention is guidance for humans, but the loader still validates semantics. A template-sweep directory must contain both `graph.template.json` and `sweep.json`. A single-graph bundle must contain one graph `.json`.

##### Example layout

```
integration_test_bundles/
  quick/
    BatchnormFwdInference/
      Inference/
        graph.template.json
        sweep.json
        golden/small_fp32_nchw/tensors.dvc
        golden/small_fp16_nchw/tensors.dvc
        golden/small_fp32_ncdhw/tensors.dvc
      nchw/
        fp32/
          miopen/
            miopen.json + miopen.tensor{0..5}.bin
    SdpaFwd/
      BhsdNoMask/
        graph.template.json
        sweep.json
        golden/small_bf16/tensors.dvc
        golden/small_fp16/tensors.dvc
  standard/
    ConvFwd/
      nhwc/
        fp16/
          resnet50_layer3/
            resnet50_layer3.json + resnet50_layer3.tensor{0..8}.bin
```

##### Filtering by generated test name

Single-graph bundles keep the legacy naming pattern: graph-derived suite plus bundle directory name. Template sweeps use a sweep-derived suite plus `cases[].id`.

| Bundle source | Generated GTest name |
|---------------|---------------------|
| `quick/BatchnormFwdInference/nchw/fp32/miopen/miopen.json` | `BatchnormFwdInference_nchw_fp32.miopen` |
| `quick/BatchnormFwdInference/Inference/sweep.json`, case `small_fp32_nchw` | `BatchnormFwdInference_Inference.small_fp32_nchw` |
| `quick/BatchnormFwdInference/Inference/sweep.json`, case `small_fp16_nchw` | `BatchnormFwdInference_Inference.small_fp16_nchw` |
| `standard/SdpaFwd/BhsdNoMask/sweep.json`, case `gqa_bf16` | `Standard/SdpaFwd_BhsdNoMask.gqa_bf16` |
| `quick/customer_issues/CASE-12345/repro/repro.json` | `ConvFwd_nchw_fp32.repro` |

**Collision handling**: Single-graph bundles collide when they produce the same suite and test name. Template-sweep bundles collide when two cases in the same sweep use the same `id`, or when two registered cases produce the same sanitized suite/test pair. Discovery rejects collisions with an error naming both sources.

Because template-sweep suite names do not include dtype, layout, dims, or feature flags, `cases[].id` is the stable filtering surface for those fields. Generators must emit ids that preserve the useful filter tokens; reviewers should treat missing dtype/layout/feature tokens as a test discoverability bug.

Filters match the generated test name:

```bash
# All batchnorm inference golden tests, single-graph or sweep
--gtest_filter=*BatchnormFwdInference*

# Batchnorm inference fp32 sweep rows
--gtest_filter=*BatchnormFwdInference*fp32*

# All legacy "typical" scenario bundles across all operations
--gtest_filter=*typical*

# All SDPA no-mask sweep cases
--gtest_filter=*SdpaFwd_BhsdNoMask*

# All conv fwd golden tests
--gtest_filter=*ConvFwd*

# All bundles testing ResNet50 shapes
--gtest_filter=*resnet50*

# Standard-tier tests only (GTest prefix)
--gtest_filter=Standard/*

# One specific template-sweep case
--gtest_filter=*BatchnormFwdInference_Inference.small_fp32_nchw

# One specific single-graph bundle
--gtest_filter=*BatchnormFwdInference_nchw_fp32.odd_spatial
```

---

### 4.2 Generation

#### Generation Pipeline

Golden data is generated by Python scripts in [`reference_data_scripts/`](../../reference_data_scripts/). The reference source is configurable per operation (see [Reference Sources](#reference-sources)); the existing batchnorm generators use PyTorch. Each generator follows the same three-step pattern:

1. **Define** -- create graph and input tensors
2. **Compute** -- run the reference source using a **per-operation precision strategy** and cast outputs to the target type. The strategy depends on the operation's rounding depth:
   - **Precision uplift** (shallow operations -- batchnorm, pointwise, activations): compute in higher precision than the target (e.g., fp64 for fp32 bundles). Few rounding events, so the trajectory divergence between reference and DUT is minimal.
   - **Matched-precision** (deep operations -- SDPA backward, multi-step chains): compute with the same intermediate precision as the DUT's accumulation contract (e.g., BF16 inputs with FP32 intermediates for a32 accumulator kernels). Higher-precision references diverge from the DUT's rounding trajectory at every stage, producing deltas dominated by accumulated precision loss rather than kernel bugs -- forcing artificially loose tolerances that hide real regressions.
3. **Write** -- save the bundle (`.json` + `.tensor{uid}.bin`) in the target precision

The generator should **auto-derive the output path** from the graph content. The developer supplies only the **bundle name**; the generator computes the directory from the graph's operation type, layout, and data type:

```bash
python generate_batchnorm_reference.py --name typical
# -> BatchnormFwdInference/nchw/fp32/typical/typical.json + .bin
```

Only batchnorm has generators and data today. Generators for the remaining operations will be added incrementally.

#### Forward-Backward Generation Constraint

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

**Architecture portability**: GPU-generated golden data may produce different numerical results across architectures, but with appropriate reference precision strategy and calibrated tolerances, cross-architecture use is possible for many operations. Architecture-independent sources (PyTorch, CPU reference) are preferred for portable regression suites. Per-architecture tagging and skipping remains available for cases where numerical differences exceed tolerance.

---

### 4.3 Validation

#### Generic Test Runner

The runner is a **base class** (`IntegrationGraphGoldenReferenceVerificationHarness`) with three concrete subclasses: `TestCpuReferenceUsingGoldenValues` (CPU reference executor), `TestGpuReferenceUsingGoldenValues` (GPU reference executor), and `IntegrationGpuGoldenReferenceEngineValidation` (GPU plugin). The base class owns discovery, loading, tolerance lookup, and comparison. Each subclass overrides the execution step. Neither subclass knows what operation a bundle contains until it loads the graph JSON at runtime.

The CPU and GPU reference subclasses validate the reference executors themselves against golden data. The engine subclass validates the engine. All three share the same base class and the same bundles — only the execution step differs.

##### How it works

For a **full single-graph bundle** (`.json` + `.bin` files):

1. **Discover** -- at GTest startup, `discoverGoldenBundles(tierDir)` recursively scans the tier directory for single-graph `.json` files. Each graph path becomes one logical bundle case -- no C++ code per test.
2. **Load** -- when a test case runs, the runner calls `loadGraphAndTensors()` which deserializes the `.json` into an executable graph object and loads the corresponding `.bin` files into tensors. If a `.meta.json` file exists, the runner reads calibrated tolerance values from it.
3. **Execute** -- run the graph through the engine under test
4. **Compare** -- check engine output against golden output from the `.bin` files using the [three-level tolerance chain](#three-levels) -- PASS or FAIL

For a **full template-sweep bundle** (`graph.template.json` + `sweep.json` + per-case golden data): discovery expands every sweep row into one logical bundle case. The expanded case builds and validates the same flatbuffer graph as a single-graph bundle before registration, then the sweep-aware loader reads golden tensor data from `golden/{CaseId}/` or the explicit `golden.path`. The loader must not use the template file's sibling path for tensors, because every case reuses the same graph template and tensor UIDs.

For a **graph-only bundle** (single graph or sweep case without golden `.bin` files -- transitional): the runner loads or expands the graph, generates inputs, runs the engine under test and a reference source, and compares their outputs at runtime.

Tolerance is looked up at runtime following the [three-level priority chain](#three-levels): TOML override -> bundle metadata (`meta.json`) -> per-operation default. No per-operation test class needed.

##### What a failure looks like

When a golden test fails, the output should give the developer everything needed to diagnose the problem without re-running or adding instrumentation:

```
FAIL: ConvFwd_nhwc_fp16.resnet50_layer3
  Bundle: quick/ConvFwd/nhwc/fp16/resnet50_layer3/resnet50_layer3.json
  Tensor: y (UID 8, output)
  Shape:  [1, 64, 56, 56]  fp16
  Max absolute error: 3.72e-03  (tolerance: 1e-03)
  Max relative error: 1.15e-02  (tolerance: 1e-02)
  Worst element: index [0, 17, 33, 41]  expected: 0.2148  actual: 0.2185
  Mismatched elements: 42 / 200704 (0.02%)
```

##### Test discovery

At GTest startup, `discoverGoldenBundles(tierDir)` recursively scans a tier directory, discovers single-graph bundles, expands template-sweep bundles into logical cases, and registers one test case per logical bundle case. Discovery is shared — the same cases are available to all runner subclasses. The [verification mode](#verification-modes) determines which runner executes each test (golden, GPU ref, CPU ref, or auto-fallback). Registration is per tier, not per operation — it never changes as new operations, bundles, or sweep rows are added.

**Decided: [`::testing::RegisterTest`](https://google.github.io/googletest/advanced.html#registering-tests-programmatically).** Each logical bundle case is registered as its own test at static initialization time. This eliminates per-operation C++ registration code and gives direct control over test naming (see [Test naming scheme](#test-naming-scheme)). `RegisterTest` applies to golden reference tests only; GPU integration tests continue to use `INSTANTIATE_TEST_SUITE_P`.

If the tier directory is empty or missing, `discoverGoldenBundles` returns an empty list (no tests, no failure). Unexpected top-level directories (e.g., `quik/` instead of `quick/`) trigger a warning to catch typos.

##### Test naming scheme

Single-graph and template-sweep bundles use different sources for the final gtest test name:

| Bundle kind | Suite source | Test source | Example |
|-------------|--------------|-------------|---------|
| Single graph | Graph content (`operation`, layout, data type) | Bundle directory name | `BatchnormFwdInference_nchw_fp32.miopen` |
| Template sweep | Sweep location (`Operation`, `TopologyName`) | Sanitized `cases[].id` | `BatchnormFwdInference_Inference.small_fp32_nchw` |

The tier prefix still comes from the top-level folder: `standard/` adds `Standard/`; `quick/` omits the prefix.

**GTest identifier validity**: All components use `[a-zA-Z0-9_]` only. The sanitizer replaces invalid characters with underscores.

**Worked examples**:

| Bundle source | Generated GTest name |
|---------------|---------------------|
| `quick/BatchnormFwdInference/nchw/fp32/miopen/miopen.json` | `BatchnormFwdInference_nchw_fp32.miopen` |
| `quick/BatchnormFwdInference/Inference/sweep.json`, case `small_fp32_nchw` | `BatchnormFwdInference_Inference.small_fp32_nchw` |
| `quick/BatchnormFwdInference/Inference/sweep.json`, case `small_fp16_nchw` | `BatchnormFwdInference_Inference.small_fp16_nchw` |
| `standard/SdpaFwd/BhsdNoMask/sweep.json`, case `gqa_bf16` | `Standard/SdpaFwd_BhsdNoMask.gqa_bf16` |
| `quick/customer_issues/CASE-12345/repro/repro.json` | `ConvFwd_nchw_fp32.repro` |

**Collision handling**: see [Collision handling](#collision-handling) above. Discovery rejects duplicate sanitized suite/test pairs with an error naming both sources. Within one template sweep, duplicate `cases[].id` values are also an error.

##### What changes from today

| Aspect | Current (batchnorm only) | This RFC |
|--------|--------------------------|----------|
| Test class | One per operation/layout/datatype | One generic class for all |
| Discovery | `getGoldenReferenceParams(subDir)` -- shallow scan, one call per directory | `discoverGoldenBundles(tierDir)` -- recursive scan plus template-sweep expansion, one call per tier |
| Adding a test | Drop files (existing op) or write C++ class (new op) | Drop a single-graph bundle or append a row to `sweep.json` plus golden data. |
| Tier assignment | GTest prefix per instantiation call | Tier folder at top level |
| Tolerance | Hard-coded per test class | Looked up from graph content and case metadata at runtime |

**Acceptance criteria**:
- [ ] Single generic test class handles all operation types
- [ ] Recursive scan discovers all single-graph bundles and expands all template-sweep cases -- no per-operation C++ code
- [ ] Adding a new test requires only dropping files in a tier folder or appending a sweep case
- [ ] Single-graph suite names derive from graph content and test names derive from bundle directory names; template-sweep suite names derive from sweep location and test names derive from sanitized `cases[].id`
- [ ] Unexpected top-level directories in golden data root produce a warning
- [ ] Empty or missing tier directory produces zero tests, not a failure
- [ ] Test name collision (two cases producing the same name) is a hard error at discovery time
- [ ] A bad bundle (malformed graph, missing `.bin` files, size mismatch) fails its own test case — other tests continue running so CI provides full signal
- [ ] An unparseable `.json` file at discovery time is skipped with a warning (no test case can be registered)
- [ ] No available reference for a bundle in `auto` mode fails the test, indicating a validation gap

#### Verification Modes

The default verification strategy is **golden-first with automatic fallback**. For each test, the runner selects the best available reference source in order:

1. **Golden data** -- if a bundle exists for this test, use it
2. **GPU reference** -- if no golden data but a GPU reference executor is available, use it
3. **CPU reference** -- last resort, if the operation is implemented

The `--verification-mode` flag overrides this default. In every mode, the **engine** is the thing being tested -- the reference source provides the expected output.

| Mode | Reference source | When reference unavailable |
|------|-----------------|---------------------------|
| `auto` (default) | Per-test fallback: golden → GPU ref → CPU ref | FAIL -- indicates a validation gap |
| `golden` | Pre-computed data from bundle | SKIP -- bundle has no golden data |
| `gpu` | GPU reference executor at runtime | SKIP -- no GPU ref for this operation |
| `cpu` | CPU reference executor at runtime | SKIP -- no CPU ref for this operation |

**Floating-point edge case**: `-0.0` vs `+0.0` uses value comparison, not bitwise. NaN handling is covered in [Data Integrity](#data-integrity).

#### Tolerance Framework

Tolerance is determined by a **three-level priority chain**. The highest-priority source that provides a value wins.

##### Three levels

| Priority | Source | Set by | When it's used |
|----------|--------|--------|----------------|
| 1 | TOML per-engine override | Engine developer | An engine needs to temporarily tighten or loosen tolerance without regenerating bundles (e.g., debugging a specific kernel, or a fused kernel that legitimately trades precision for throughput) |
| 2 | Bundle tolerance (`meta.json`) | Generation/calibration tool | Steady state -- each bundle carries its own calibrated tolerance, set at generation time by measuring expected variance |
| 3 | Per-operation default (code) | Compile-time constant | Safety net for bundles without calibrated tolerance (e.g., bundles migrated from the old test system) |

If a TOML override matches, it wins. Otherwise the bundle's `meta.json` tolerance applies. If the bundle has no tolerance metadata, the per-operation default applies.

The TOML mechanism already exists. Each engine's config (e.g., [`MIOPEN_ENGINE.toml`](../../../../dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml)) can declare `[[tolerance_overrides]]` entries with glob-pattern filters and atol/rtol values.

##### Bundle tolerance (level 2)

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

- **This RFC -- fixed tolerances**: The runner reads the operation type and data type from the graph JSON and uses them as a lookup key into [`TestTolerances.hpp`](../../test_sdk/include/hipdnn_test_sdk/utilities/TestTolerances.hpp), which defines compile-time constants per operation and data type (e.g., batchnorm inference fp32 = `2e-4`, conv fwd fp32 = `1e-5`). The graph does not compute tolerances — it provides the key, the lookup table provides the value.

- **Future -- dynamic tolerances**: [`DynamicTolerances.hpp`](../../test_sdk/include/hipdnn_test_sdk/utilities/DynamicTolerances.hpp) will replace fixed lookup values with tolerances computed from graph properties (operation, data type, tensor dimensions). This is the intended long-term solution but requires validation before it can replace fixed values (see [Future Work](#future-work)).

This evolution changes *how* the default is computed -- it does not change the three-level chain.

##### Current gap

The golden ref framework (`TestCpuReferenceUsingGoldenValues` / `TestGpuReferenceUsingGoldenValues` / `IntegrationGpuGoldenReferenceEngineValidation`) takes tolerances as hard-coded function parameters -- it bypasses both the TOML override lookup and the bundle metadata lookup. Golden tests must be connected to the three-level flow: check TOML override first, then bundle metadata, then fall back to per-operation default.

**Acceptance criteria**:
- [ ] Golden tests follow the three-level tolerance chain: TOML override → bundle metadata → per-operation default
- [ ] Per-engine TOML overrides apply to golden tests -- no separate override mechanism
- [ ] Generation tool writes calibrated tolerance to `meta.json` using the empirical calibration procedure
- [ ] Bundles without tolerance metadata fall back to per-operation default (no crash, no skip)

#### Data Integrity

Internal consistency is guaranteed by construction for generated bundles: `Graph.save()` writes single-graph JSON and `.bin` files from the same in-memory graph in a single call, so UIDs and tensor data correspond. Template-sweep tooling must provide the same guarantee after expansion: each row in `sweep.json` expands to a valid graph, and tensor data is loaded from that case's `golden/{CaseId}/` directory or explicit `golden.path`, never from the template directory. Corruption can only enter after generation -- partial downloads, disk errors, manual edits, or incorrect sweep rows.

Three checks catch the real tensor-data failure modes:

1. **Tensor size validation (load time)** -- *proposed*. Verify that the loaded byte count matches `element_space x sizeof(element_type)` declared in the expanded graph JSON (see [binary tensor format](#binary-tensor-format)). Not performed today -- a truncated file silently produces garbage. This is the primary load-time integrity check -- fast and sufficient to catch truncated downloads and wrong-file swaps.

2. **NaN/Inf rejection (generation time)** -- *proposed*. Reject output tensors containing NaN or Inf before writing any files. All-same-value tensors are valid (e.g., a bias-only layer can produce uniform output).

3. **NaN/Inf rejection (pre-commit)** -- *proposed*. Safety net for bundles generated by external tools or before check #2 is added. The **pre-commit bundle verifier** scans output tensors and rejects any containing NaN or Inf.

The **pre-commit bundle verifier** runs checks #1-#3 across a directory tree before bundles are committed. DVC content-addressing handles file integrity, so no separate checksum is needed. The verifier also validates structural conventions and data-size budgets:

4. **Tier folder validation** -- warn about top-level directories that are not one of the four tier names
5. **Single-graph JSON validation** -- verify each single-graph `.json` file is a parseable graph (not a stray `README.json` or editor config)
6. **Template-sweep validation** -- verify `graph.template.json`, `sweep.json`, placeholder coverage, per-case tensor UID/dims/strides/data-type coverage, duplicate ids, golden DVC pointer paths, and expanded graph schema validity
7. **Missing `.bin` files** -- for full single-graph bundles and expanded template cases, verify that every tensor UID in the graph has a corresponding `.bin` file in that bundle or case directory
8. **Forward-backward consistency** -- for backward bundles, verify that `meta.json` includes a `forward_source` field (see [Forward-Backward Generation Constraint](#forward-backward-generation-constraint))
9. **DVC payload budget** -- warn when one expanded case's tensor payload exceeds 2 MB, report aggregate tensor payload for the tree, and fail when committed golden tensor payload exceeds the 800 MB project budget from [Bundle size considerations](#bundle-size-considerations)

**Acceptance criteria**:
- [ ] All checks implemented with actionable error messages naming the file and tensor UID
- [ ] File size mismatch and NaN/Inf in golden data are hard FAILs, not warnings
- [ ] Pre-commit bundle verifier validates full and graph-only bundles before commit
- [ ] Pre-commit and CI bundle verification report per-case and aggregate tensor payload and enforce the warning/fail thresholds from [Bundle size considerations](#bundle-size-considerations)
- [ ] Stray non-graph `.json` files and unexpected top-level directories produce warnings

---

### 4.4 CLI and Configuration

`--verification-mode` controls **what** runs. `--golden-data-dir` controls **where** golden data is read from (ignored when mode is `cpu` or `gpu`).

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--vm, --verification-mode` | `auto`, `golden`, `gpu`, `cpu` | `auto` | Which reference source to use (see [Verification Modes](#verification-modes)) |
| `--gd, --golden-data-dir` | path | `<exe_dir>/../lib/integration_test_bundles` | Where to find bundle data (only used when mode includes golden tests) |

Each flag has an environment variable fallback. The CLI flag takes precedence when both are set.

| Flag | Environment variable |
|------|---------------------|
| `--verification-mode` | `HIPDNN_TEST_VERIFICATION_MODE` |
| `--golden-data-dir` | `HIPDNN_TEST_GOLDEN_DATA_DIR` |

**Acceptance criteria**:
- [ ] Both flags parsed and stored in `TestConfig` singleton
- [ ] Environment variable fallbacks work when CLI flag is absent
- [ ] `--verification-mode golden` skips tests that have no golden data
- [ ] `--verification-mode gpu` and `--verification-mode cpu` ignore golden data entirely (no fetch, no directory check)

---

## Integration

### CI Integration

Tiers are determined by the top-level folder (see [Folder Convention](#folder-convention)), following the same [tier cascade](../../../../dnn-providers/integration-tests/README.md#how-tiers-cascade) as all other integration tests.

| CI Stage | ctest Command | Verification Mode | Notes |
|----------|--------------|-------------------|-------|
| Pre-submit (smoke) | `ctest -L quick` | `auto` | `quick/` tier golden tests; tests without golden data fall back to GPU/CPU ref |
| Post-submit | `ctest -L standard` | `auto` | `quick/` + `standard/` tier golden tests; fallback to computed for gaps |
| Nightly | `ctest -L comprehensive` | `golden` | All tiers up to comprehensive; tests without golden data are skipped |
| Weekly | `ctest -L full` | `golden` | All tiers |

### Workflows

**Add a test -- any operation, any data type** (developer):
1. Write a generation script following the [`generate_batchnorm_reference.py`](../../reference_data_scripts/generate_batchnorm_reference.py) pattern, run it to produce a bundle
2. Run the pre-commit bundle verifier to verify the bundle is well-formed
3. Drop the bundle folder into the appropriate tier directory (e.g., `integration_test_bundles/quick/ConvFwd/nhwc/fp16/MyTest/`) and commit (DVC for large tensors). No C++ changes, no recompile -- the generic runner discovers it automatically

**Debug a customer issue** (support):
1. Receive the customer's bundle folder -- no source code or NDA required
2. Drop the folder into any tier directory (e.g., `integration_test_bundles/quick/customer_issues/CustomerBundle/`)
3. Run tests -- the runner picks it up, executes the engine, compares against golden output
4. Inspect the diff: which tensors diverge, by how much, at which indices

**Reproduce a CI failure locally** (developer):
1. Pull the golden data via DVC
2. Run the failing test with `--gtest_filter=*OperationName*DataType*`
3. The bundle is self-contained -- no environment-specific setup beyond the engine under test

**Add golden test support for a new plugin** (provider developer):
1. Subclass `IntegrationGraphGoldenReferenceVerificationHarness` in your provider's tests directory -- override the execution step with your plugin's setup/teardown
2. The same bundles are automatically validated against your plugin -- no new test data needed

---

## Migration Plan

The transition from per-operation test classes to the generic runner is incremental. Old and new coexist at every phase -- no flag day.

| Phase | What ships | Coexistence |
|-------|-----------|-------------|
| 1 | Generic runner (`IntegrationGraphGoldenReferenceVerificationHarness` + subclasses), discovery, tier folders. Migrate existing batchnorm bundles to `integration_test_bundles/quick/`. | Old per-operation classes (`TestCpuFpReferenceBatchnorm`, etc.) remain, running the same bundles. Both suites run in CI. |
| 2 | Connect three-level tolerance chain (TOML override -> bundle metadata -> per-operation default). Add pre-commit bundle verifier. | Old classes still active. Golden tests now use the same tolerance path as computed tests. |
| 3 | Generate bundles for conv, matmul, and pointwise. Validate generic runner covers these operations end-to-end. | Old batchnorm classes can be removed once the generic runner has run green for a full release cycle. |
| 4 | CI integration: DVC pull, per-tier minimum test count baseline, `--verification-mode auto` as default. | Old computed-only CI jobs unchanged. Golden CI jobs added alongside. |
| 5 | Expand to remaining operations (SDPA, layernorm, RMS norm, reduction). Remove remaining per-operation golden test classes. SDPA onboarding requires the matched-precision reference strategy, the forward-backward generation constraint, empirically calibrated tolerance for non-deterministic engine variants, and the tolerance calibration tool — see the relevant sections in Detailed Design. | Generic runner is the sole golden test path. Computed tests (`IntegrationGraphVerificationHarness`) remain for operations without golden data. |

**Rollback**: At any phase, setting `--verification-mode cpu` (or `gpu`) disables golden tests entirely. Old per-operation classes are not removed until the generic runner has proven stable for that operation.

---

## Data Management

Golden data lives in two places -- **source tree** and **runtime**:

- **Source tree**: Bundle data lives at `dnn-providers/integration-tests/integration_test_bundles/`. The existing batchnorm data currently lives at `projects/hipdnn/hipdnn_reference_data/` and will be moved here.

- **Runtime**: The test binary reads golden data directly from the source tree -- no CMake copy step. The default path is resolved relative to the executable. The `--golden-data-dir` CLI flag or `HIPDNN_TEST_GOLDEN_DATA_DIR` env var overrides this location.

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

Golden tensor payload under `dnn-providers/integration-tests/integration_test_bundles/` should remain below 1 GB total. To leave headroom, the bundle verifier should warn when any individual expanded case exceeds 2 MB and fail when the aggregate committed tensor payload exceeds 800 MB. Cases larger than the per-case warning need an explicit rationale in review. When tensors are reproducible from `(shape, dtype, layout, mask/features, seed)` and static data is not needed to reproduce a bug, prefer a graph-only bundle with runtime generation over committing large `.bin` payloads.

##### Shape selection policy

Generators should produce a **representative subset** of shapes per operation, not an exhaustive matrix. The goal is to cover tolerance-critical corners:

- **Small shapes**: where relative error dominates (e.g., single-element edge cases)
- **Medium shapes**: typical production workloads
- **Large shapes**: where accumulation length stresses the math (e.g., long sequence SDPA)

Not every combination of batch × heads × sequence × dim is needed. Select shapes that exercise distinct numerical regimes.

##### Tier placement

Tier placement is governed by **runtime budget**, not bundle size, following [TheRock's test filtering model](https://github.com/ROCm/TheRock/blob/main/docs/development/test_filtering.md#types-of-filters). A small bundle with an expensive engine may belong in `standard/`; a large bundle with a fast engine may fit in `quick/`. The auto-tier classifier should time test execution and suggest placement accordingly.

##### DVC pull at CI time

DVC is already in the repo for other large binary assets, content-addressing provides integrity guarantees, and selective fetch by path avoids pulling data for operations not under test. CI jobs run `dvc pull integration_test_bundles/{tier}/` before test execution to fetch only the data needed for the tier under test. Developers can fetch specific operations: `dvc pull integration_test_bundles/quick/batchnorm*/` without pulling unrelated multi-MB bundles. Fallback: `--verification-mode cpu` (or `gpu`) runs without bundle data if DVC is unavailable, so CI is never fully blocked by a storage outage.

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| FlatBuffers schema change | Old JSON bundles unreadable by `loadGraphAndTensors()` | Low | Schema changes must be backwards compatible -- old graphs continue to work when new fields are added. FlatBuffers supports this natively via optional fields |
| Reference script bug freezes wrong data | Silent incorrect baseline | Medium | Cross-validate against C++ CPU ref; review generated data before committing; [proposed generation-time validation](#data-integrity) will reject degenerate outputs |
| PyTorch version drift | Different versions produce slightly different outputs | Low | Pin PyTorch version in `requirements.txt`; regenerate when upgrading |
| Large golden data sets slow CI | CI feedback loop degrades | Low | Storage caching, selective fetch by test filter, per-case and aggregate DVC payload budget checks, and graph-only/runtime generation for reproducible large cases |
| Remote storage unavailable | Golden-mode CI fails | Low | Computed-mode CI is independent of storage; CI fallback to computed-only |
| Silent coverage regression | Accidental bundle deletion reduces test count; CI stays green | Low | Code review; DVC history tracks all bundle changes |
| Non-deterministic engine flakiness | Golden comparison fails intermittently for engines using non-deterministic accumulation (e.g., atomic FP32 additions where thread-block scheduling varies run-to-run) | Medium | Tolerance must cover DUT run-to-run variance, calibrated empirically from repeated runs -- not hand-tuned. `deterministic` metadata on bundles aids triage but is not the mitigation. CI must exercise the production engine path, not a deterministic substitute |
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

- **Graph-derived names vs. path-derived names vs. explicit case ids.** Path-derived names are stable against graph edits but break the customer-bundle workflow and make compressed sweeps depend on generated paths. Pure graph-derived names are stable against folder reorganization but do not distinguish many same-topology sweep rows. *Why we chose hybrid explicit identity: single-graph bundles use graph-derived suites plus bundle directory names, while template sweeps use sweep-derived suites plus explicit `cases[].id`. Collisions are detected at discovery time with a clear error.*

- **DVC vs. Git LFS vs. pre-staged CI artifact.** DVC is already in the repo for other large assets; LFS is more universal but adds a new dependency; a pre-staged artifact decouples test latency from data fetch but loses content-addressing. *Why we chose DVC: already in the repo for other large assets, content-addressing provides integrity guarantees, and selective fetch by path avoids pulling data for operations not under test.*

- **Per-test checksum verification vs. DVC content-addressing.** Per-tensor SHA-256 checksums stored in JSON would catch corruption at any stage, but DVC already content-addresses every tracked file. *Why we chose DVC: already in the repo, no extra metadata to maintain, and tensor size validation at load time catches the practical failure modes (truncation, wrong file).*

- **Tier as top-level folder vs. tier as bundle metadata.** Folder-as-tier maps cleanly to `ctest -L`, but moving a bundle between tiers requires `git mv`. Metadata-as-tier keeps location stable but requires reading every JSON to build the tier list. *Why we chose folder-as-tier: maps directly to `ctest -L` without reading any JSON, matches how the rest of the integration suite assigns tiers, and makes tier membership visible in the directory listing.*

---

## Future Work

1. **Dynamic tolerance integration**: Wire the existing `DynamicTolerances` functions (matmul, conv, batchnorm, layernorm, RMS norm, pointwise) into the golden test runner so a single generic test class handles all operations.
2. **Engine-as-bundle-producer**: Any engine can take a graph-only bundle, execute the graph, and write the results back as a full bundle -- enabling cross-engine comparison without a shared reference source.
3. **Bundle-to-bundle comparison**: A standalone tool that loads two bundles and diffs their output tensors directly, matched by graph content rather than filename.
4. **Reproducible generation**: Fixed seeds for random input generation so that regenerating a bundle produces the same inputs, isolating output differences to the reference source change.
5. **Auto-tier classification**: The generator suggests the appropriate tier folder based on tensor element counts, matching the existing size conventions.
6. **KnobSettings coverage**: Validate the same golden bundles under different engine KnobSettings configurations to catch regressions from tuning changes.
7. **Tolerance calibration tool**: Extend the generation framework to run the target engine N times on each bundle, compute empirical p99.9 tolerance bounds, and write calibrated tolerance values to `meta.json`. Required before onboarding operations with non-deterministic engines (e.g., SDPA backward with atomic accumulation).
8. **Structural template-sweep composition**: If several template sweeps become near-duplicates because their graphs differ only by optional tensors, node inputs, or node outputs (for example SDPA stats, FP8 descale tensors, or GROUP sequence-length tensors), extend `graph.template.json` and `sweep.json` with an explicit structural templating mechanism so those sweeps can be combined. This is intentionally out of v1 until real duplication justifies the extra schema and validation complexity.
