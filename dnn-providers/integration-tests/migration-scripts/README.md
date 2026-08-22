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
VERIFIED at byte level — the migrated graph is identical to the captured one
    |
    |  Hop D: diff_coverage.py (run both suites, compare pass sets)
    v
VERIFIED at behavior level — every C++ PASS has a bundle that also PASSes
```

**Hop C proves the bytes survived; Hop D proves the behavior survived.** A
graph can round-trip perfectly on disk yet fail to run — Hop D is what makes
"turn off the C++ tests, lose nothing" a checked fact rather than a hope.

### Quick path: convert one existing C++ test (no full pipeline needed)

Hops B–D (`place_bundles.py`, `verify_migration.py`, `diff_coverage.py`) exist
to migrate a whole suite in bulk with a byte- and behavior-level proof. For
**one** test, skip them and go straight from capture to import — two steps:

```bash
# Step 1: Capture — dump the graph(s) for just this test
./build/bin/hipdnn_integration_tests --capture-bundles /tmp/captured \
    --gtest_filter='Full/IntegrationGpuConvFwdBiasActiv2dFp16.Correctness/*'

# Step 2: Import — merge each captured graph into the bundle tree
for graph in /tmp/captured/*/*/*.json; do
    [[ "$graph" == *.meta.json ]] && continue
    python3 migration-scripts/import_graph.py \
        --graph "$graph" \
        --bundle-dir dnn-providers/integration-tests/integration-test-bundles \
        --meta reference_source="c++ integration suite: $(basename "$(dirname "$graph")")"
done
```

`import_graph.py` is idempotent and dedup-aware (see Step 5 below): it
appends a new sweep case if a matching topology already exists, creates a
new template+sweep if not, and reports `DUPLICATE` if the exact case is
already bundled. Verify each printed `--gtest_filter` line runs and passes
as a bundle case, then delete the C++ `TEST_P`/`INSTANTIATE_TEST_SUITE_P`
registration for that case.

**This produces a graph-only bundle** — `--capture-bundles` dumps the graph
topology only, never golden output tensors, so `import_graph.py` has nothing
to attach as `.tensors.dvc`/`.bin`. Golden data is optional (see [RFC 0011
§4.1](../../../projects/hipdnn/docs/rfcs/0011_GoldenReferenceValidation.md));
a graph-only case still runs and is verified against the GPU/CPU reference
executor. Generate and commit golden tensors separately (a per-op generator
script, see `integration-test-bundles/README.md`) only if you want the more
sensitive golden-comparison mode for that case too.

**Note:** `import_graph.py` does not read the `.meta.json` sidecar that
`--capture-bundles` writes next to each graph — pass `--seed` and
`--meta inputs=<json>` explicitly (copy the values out of the `.meta.json`)
if you need the imported case's seed/input-range metadata to match the
original C++ test exactly.

Use the full pipeline below instead when migrating many tests/suites at
once and you want the Hop C/D byte- and behavior-level proof that nothing
was lost.

## How to Run

### Prerequisites

Build the integration test binary (no GPU required for Layers 1-3):

```bash
cmake --build build --target hipdnn_integration_tests
```

### Automated: Full pipeline

```bash
migration-scripts/run_capture_pipeline.sh build/bin/hipdnn_integration_tests
```

This runs Hops A–D plus the supporting checks in sequence. Pass
`--skip-hopd` to skip the differential coverage check (which requires a GPU).

### Manual: Step by step

#### Step 1: Census — see what exists

```bash
python3 migration-scripts/census.py build/bin/hipdnn_integration_tests
```

Runs `--gtest_list_tests` and classifies every test case as `graph`
(C++ graph test to migrate), `bundle` (already a bundle), or `other`.

#### Step 2: Capture (Hop A) — serialize C++ graphs as JSON

```bash
./build/bin/hipdnn_integration_tests --capture-bundles captured_bundles \
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
python3 migration-scripts/place_bundles.py \
    --capture-dir captured_bundles \
    --output-dir dnn-providers/integration-tests/integration-test-bundles
```

Groups captured graphs by **structure** (node types + wiring + tensor
set). Graphs sharing the same topology collapse into one template+sweep.
Use `find_case.py` to query cases by any parameter (see below).

#### Step 4: Verify (Hop C) — reconcile everything

```bash
python3 migration-scripts/verify_migration.py \
    --census census.json \
    --capture-dir captured_bundles \
    --bundle-dir integration-test-bundles
```

Three-way reconciliation: census count == captured count == placed count,
plus per-case byte-exact comparison of graph AND metadata (seed, inputs).

#### Step 5: Import individual graphs (incremental)

```bash
python3 migration-scripts/import_graph.py \
    --graph path/to/graph.json \
    --bundle-dir integration-test-bundles/ \
    --meta reference_source="c++ integration suite: Suite.Case"
```

Dedup-aware placement. Default: skip exact duplicates. `--strict` exits
non-zero on dup (CI mode). `--force` appends regardless.

## Searching and Running Bundles

### find_case.py — query cases by any parameter

`find_case.py` reads `sweep.json` directly (single source of truth).
No separate manifest to maintain.

```bash
# List all batchnorm cases
python3 migration-scripts/find_case.py --op Batchnorm

# Find fp16 nhwc cases
python3 migration-scripts/find_case.py --dtype fp16 --layout nhwc

# Find cases that have an epsilon input (any range)
python3 migration-scripts/find_case.py --input epsilon

# Find cases where epsilon is in [-1,1]
python3 migration-scripts/find_case.py --input epsilon:-1,1

# Find by shape
python3 migration-scripts/find_case.py --shape 1x16x3x3

# Combine filters
python3 migration-scripts/find_case.py --op Batchnorm --dtype bfp16 --input scale:-2,2

# Full detail for a case (includes the exact --gtest_filter command)
python3 migration-scripts/find_case.py --id f446b9 --detail
```

### gtest_filter examples

Bundle suites register under gtest as `{tier}_{Op}_{Topology}` (e.g.
`quick_Batchnorm_Default`). Each case within a suite is named by its
case id:

```bash
# Run all quick-tier bundles
--gtest_filter='quick_*'

# Run all batchnorm cases (any tier)
--gtest_filter='*Batchnorm*'

# Run all bfp16 nhwc cases across all ops
--gtest_filter='*bfp16_nhwc*'

# Run one exact case (by hash suffix)
--gtest_filter='*f446b9*'

# Combine: quick batchnorm, only fp32
--gtest_filter='quick_Batchnorm_*/*fp32*'
```

When two cases share the same shape/dtype/layout but differ in input
ranges or seeds, a 6-char content hash is appended to disambiguate:

```
1_3_14_bfp16_ncl          ← unique readable prefix, no hash needed
1_3_14_bfp16_ncl_abc123   ← same prefix, different input ranges
1_3_14_bfp16_ncl_def456   ← same prefix, different input ranges
```

Use `find_case.py --id abc123 --detail` to see what a hashed case contains.

### Adding new test cases

New tests should be added directly as bundle cases — no C++ needed.

**Using import_graph.py (recommended):**

```bash
python3 migration-scripts/import_graph.py \
    --graph new_conv.json \
    --bundle-dir integration-test-bundles/
```

What happens:

1. The script computes the graph's skeleton hash and finds matching
   topologies in the bundle tree.
2. **Duplicate?** If an identical case already exists (same graph +
   seed + inputs), it reports `DUPLICATE` and skips. No manual check
   needed.
3. **New case for existing topology?** Appends to that sweep. The case
   id is auto-generated: `{shape}_{dtype}_{layout}_{attrs}[_{hash6}]`.
4. **New topology?** Creates a new template+sweep directory.
5. The auto-generated id is printed to stderr so you see what the test
   will be called in gtest output:

```
  appended case '1_16_3_3_bfp16_nhwc_dil1x1_prepad1x1_postpad1x1' to
    integration-test-bundles/quick/ConvolutionFwd/Default/sweep.json
```

That case id is the gtest name — it appears in CI logs, `--gtest_filter`,
and `find_case.py` queries. You never need to invent or assign it.

**Manually editing sweep.json:**

Add a new entry to the `"cases"` array with `"values"` and `"metadata"`.
Set `"id"` to a descriptive name following the pattern
`{shape}_{dtype}_{layout}[_{attrs}]`, or run `import_graph.py` to have
it assigned automatically.

The migration pipeline (`run_capture_pipeline.sh`) is a one-time
conversion tool. Going forward, the bundle tree is the source of truth.

## Verification

Two hops prove the migration; two supporting checks guard the tooling:

| Check | What | Needs GPU? |
|---|---|---|
| **Hop C — byte round-trip** | Census ↔ capture ↔ sweep count reconciliation, per-case byte-diff of graph + seed + inputs | No |
| Supporting — loader smoke | Real C++ binary loads every placed bundle via the production template expander | Depends on test |
| Supporting — idempotency | Re-run pipeline; `git diff --exit-code` must be clean | No |
| **Hop D — diff coverage** | `pass_set_bundle ⊇ pass_set_cpp` — no C++ PASS becomes a bundle SKIP | **Yes** |

Hop D is the acceptance proof: when it's green, turning off C++
integration tests provably loses no coverage for graph+GPU-plugin tests.

```bash
python3 migration-scripts/diff_coverage.py \
    --cpp /tmp/cpp.json --bundle /tmp/bundle.json \
    --bundle-dir integration-test-bundles
```

## Field Mapping — What Gets Bundled and Where

| Field | C++ Origin | Bundle Destination | Compression |
|---|---|---|---|
| Node types + wiring | `buildGraph()` | `graph.template.json` nodes[] | Invariant (skeleton) |
| Tensor dims | `TensorAttributes` | sweep `values.tensors[uid].dims` | Per-case |
| Tensor strides | `generateStrides()` | sweep `values.tensors[uid].strides` | Per-case |
| data_type | `getDataTypeEnumFromType<T>()` | sweep `values.tensors[uid].data_type` | Per-case |
| Node attrs | `ConvFpropAttributes` etc. | sweep `values.attributes.<key>` | Per-case, only if varies |
| Seed | `inputFillRecipes().setGlobalSeed()` | sweep `case.metadata.seed` | Per-case metadata |
| Distribution/range | `InputFillRecipes.fills()` | sweep `case.metadata.inputs.{uid}` | Per-case metadata |
| Per-op defaults | `FillInputs.cpp` | **NOT stored** | Re-derived from topology |
| Provenance | (new) | sweep `case.metadata.reference_source` | Per-case metadata |

## Scripts

| Script | Purpose |
|---|---|
| `bundle_utils.py` | Shared utilities (skeleton hash, canonicalization, expansion, case-id) |
| `census.py` | List and classify all C++ test cases (the migration denominator) |
| `place_bundles.py` | Convert captured bundles into template+sweep format (Hop B) |
| `verify_migration.py` | Reconcile census ↔ capture ↔ sweep, byte-diff graphs + metadata (Hop C) |
| `import_graph.py` | Import a single graph with duplicate detection |
| `find_case.py` | Query cases by op, dtype, layout, shape, input range, or id |
| `diff_coverage.py` | Differential coverage: assert `pass_set_bundle ⊇ pass_set_cpp` (Hop D) |
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

### Case ids — handle, not description

A sweep case can differ along many axes: tensor dims, strides, dtype, node
attributes (padding/stride/dilation), and per-tensor input ranges/seeds. That
is too high-dimensional to encode in a readable gtest name. So the id is a
**handle**, following the content-addressable pattern (`docker`, `git`, `nix`):

```
{shape}_{dtype}_{layout}_{≤3 salient attrs}[_{hash6}]
```

- The readable tokens (shape, dtype, layout, top attrs) are the **filter
  surface** (RFC 0011 §4.1) — they only encode axes that actually vary in the
  sweep, so ids stay short.
- The `hash6` suffix is a short content hash over the case's **full identity**
  (`values` + `metadata.inputs` + `seed`), appended only when the readable base
  is not already unique. It guarantees uniqueness — including for two cases that
  differ *only* in an input range (e.g. a bias filled from `[-0.5, 0.5]` vs
  `[-1, 1]`) — and is stable across additions/reordering, unlike an `_N` counter.

Use `find_case.py` to query the full per-case truth the id cannot carry.

### Verify gate

Every expanded case is compared byte-for-byte against the original.
Any mismatch falls back to a standalone bundle. Nothing is silently dropped.
