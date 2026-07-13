# Migration Scripts — C++ Graph Tests to Bundle Format

These scripts convert C++ integration graph tests into the compressed
template+sweep bundle format (ALMIOPEN-2221, ALMIOPEN-2279).

## Why

C++ graph tests build their graphs programmatically — each test is a
`buildGraph()` function plus `INSTANTIATE_TEST_SUITE_P` with hardcoded
parameter lists. This makes them hard to extend, audit, and reuse.

The bundle format stores graphs as JSON, decoupling the test data (graph
topology, tensor shapes, dtypes) from the test harness (build, execute,
verify). A single template+sweep pair can replace dozens of C++ test
registrations that only differ in shapes/dtypes/layouts.

## Pipeline

```
C++ graph test
    |
    |  Hop A: --capture-bundles (run the binary, serialize each graph)
    v
standalone bundle (one JSON per test case)
    |
    |  Hop B: place_bundles.py (group by structure, templatize, compress)
    v
template + sweep (one template per topology, one sweep with all cases)
    |
    |  Hop C: verify_migration.py (reconcile counts + byte-diff graphs/metadata)
    v
VERIFIED — lossless migration proven
```

## How to Run

### Prerequisites

Build the integration test binary (no GPU required for Layers 1-3):

```bash
cmake --build build --target integration_tests
```

### Automated: Full pipeline

```bash
migration_scripts/run_capture_pipeline.sh build/bin/integration_tests
```

This runs all four verification layers in sequence. Pass `--skip-layer4`
to skip the differential coverage check (which requires a GPU).

### Manual: Step by step

#### Step 1: Census — see what exists

```bash
python3 migration_scripts/census.py build/bin/integration_tests
```

Runs `--gtest_list_tests` and classifies every test case as `graph`
(C++ graph test to migrate), `bundle` (already a bundle), or `other`.

#### Step 2: Capture (Hop A) — serialize C++ graphs as JSON

```bash
./build/bin/integration_tests --capture-bundles captured_bundles \
    --gtest_filter='*IntegrationGpu*'
```

The `--gtest_filter` restricts capture to `IntegrationGpu*` graph tests
— the tests that build a graph and verify it across GPU plugins. These
are the tests whose graphs can be bundled and re-executed with any
plugin. Non-graph tests (e.g. perf benchmarks, serialization tests)
use different naming prefixes and are excluded by construction.

Each C++ graph test serializes its graph JSON and metadata into:
```
captured_bundles/{SuiteName}/{CaseName}/{CaseName}.json
captured_bundles/{SuiteName}/{CaseName}/{CaseName}.meta.json
```

#### Step 3: Place (Hop B) — compress into template+sweep

```bash
python3 migration_scripts/place_bundles.py \
    --capture-dir captured_bundles \
    --output-dir dnn-providers/integration-tests/integration_test_bundles
```

Groups captured graphs by **structure** (node types + wiring + tensor
set). Graphs sharing the same topology collapse into one template+sweep.

#### Step 4: Verify (Hop C) — reconcile everything

```bash
python3 migration_scripts/verify_migration.py \
    --census census.json \
    --capture-dir captured_bundles \
    --bundle-dir integration_test_bundles
```

Three-way reconciliation: census count == captured count == placed count,
plus per-case byte-exact comparison of graph AND metadata (seed, inputs).

#### Step 5: Import individual graphs (incremental)

```bash
python3 migration_scripts/import_graph.py \
    --graph path/to/graph.json \
    --bundle-dir integration_test_bundles/ \
    --meta ported_from="c++ integration suite: Suite.Case"
```

Dedup-aware placement. Default: skip exact duplicates. `--strict` exits
non-zero on dup (CI mode). `--force` appends regardless.

## Verification Layers

Four layers, escalating in strength:

| Layer | What | Needs GPU? |
|---|---|---|
| **1 — Hop C** | Census + capture + sweep count reconciliation, per-case byte-diff of graph + seed + inputs | No |
| **2 — Smoke** | Real C++ binary loads every placed bundle via the production template expander | Depends on test |
| **3 — Idempotency** | Re-run pipeline; `git diff --exit-code` must be clean | No |
| **4 — Diff coverage** | `pass_set_bundle ⊇ pass_set_cpp` — no C++ PASS becomes a bundle SKIP | **Yes** |

Layer 4 is the acceptance proof: when it's green, turning off C++
integration tests provably loses no coverage for graph+GPU-plugin tests.

```bash
python3 migration_scripts/diff_coverage.py \
    --cpp /tmp/cpp.json --bundle /tmp/bundle.json \
    --bundle-dir integration_test_bundles
```

## Field Mapping — What Gets Bundled and Where

| Field | C++ Origin | Bundle Destination | Compression |
|---|---|---|---|
| Node types + wiring | `buildGraph()` | `graph.template.json` nodes[] | Invariant (skeleton) |
| Tensor dims | `TensorAttributes` | sweep `values.tensors[uid].dims` | Per-case |
| Tensor strides | `generateStrides()` | sweep `values.tensors[uid].strides` | Per-case |
| data_type | `getDataTypeEnumFromType<T>()` | sweep `values.tensors[uid].data_type` | Per-case |
| Node attrs | `ConvFpropAttributes` etc. | sweep `values.attributes.<key>` | Per-case, only if varies |
| Seed | `synthesis().setGlobalSeed()` | sweep `case.metadata.seed` | Per-case metadata |
| Distribution/range | `SynthesisConfig.fills()` | sweep `case.metadata.inputs.{uid}` | Per-case metadata |
| Per-op defaults | `SynthesizeInputs.cpp` | **NOT stored** | Re-derived from topology |
| Provenance | (new) | sweep `case.metadata.ported_from` | Per-case metadata |

## Scripts

| Script | Purpose |
|---|---|
| `bundle_utils.py` | Shared utilities (skeleton hash, canonicalization, expansion, case-id) |
| `census.py` | List and classify all C++ test cases (the migration denominator) |
| `place_bundles.py` | Convert captured bundles into template+sweep format (Hop B) |
| `verify_migration.py` | Reconcile census ↔ capture ↔ sweep, byte-diff graphs + metadata (Hop C) |
| `import_graph.py` | Import a single graph with duplicate detection |
| `diff_coverage.py` | Differential coverage: assert `pass_set_bundle ⊇ pass_set_cpp` (Layer 4) |
| `run_capture_pipeline.sh` | Orchestrate all hops + verification layers |
| `test_migration.py` | Self-test on synthetic fixture (no binary needed) |

## How It Works

### Structure-hash grouping

Every graph gets a canonical skeleton fingerprint: node types in order,
tensor UID wiring (canonically renumbered), and the tensor set. Graphs
with the same fingerprint share a topology.

### Derive, don't classify

Instead of maintaining a per-op allowlist:
- **Structural** (fixed in template): `node.type`, `*_tensor_uid`,
  `tensor.uid`, `tensor.virtual`
- **Knob** (templatized if varies): dims, strides, dtypes, node attrs

### Verify gate

Every expanded case is compared byte-for-byte against the original.
Any mismatch falls back to a standalone bundle. Nothing is silently dropped.
