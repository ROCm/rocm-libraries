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
**one** test, capture it and import each graph with `import_graph.py`. The
commands, and the metadata and support-sidecar steps that a finished bundle
needs, are in [Adding Tests — From an existing C++ graph test](../docs/adding-tests.md#from-an-existing-c-graph-test).

Use the full pipeline below instead when migrating many tests/suites at
once and you want the Hop C/D byte- and behavior-level proof that nothing
was lost. The pipeline writes only graphs and case metadata. Every bundle that
it places still needs a support sidecar (see
[Updating support claims](../docs/adding-tests.md#updating-support-claims)).

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

Writing over an existing tree keeps the hand-written cases of each existing
`sweep.json` (cases whose metadata has `"generator": "manual"`, such as the
runtime pass-by-value cases). No capture reproduces them. If one no longer fits
the regenerated template, the run fails and leaves that sweep unchanged; move or
fix the case by hand.

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
    --graph captured_bundles/Suite/Case/Case.json \
    --bundle-dir integration-test-bundles/
```

Dedup-aware placement. Default: skip exact duplicates. A case whose tensor
lacks a field that the template's tensor has, where the template field is not a
`${case.*}` placeholder, gets a `tensor_patches` entry that removes the field.
Metadata and the tier
come from the graph's `.meta.json` sidecar; an unusable sidecar fails the
import. `--strict` (CI mode) also exits non-zero on a duplicate or on a
`--seed`/`--meta`/`--tier` value that contradicts the sidecar. `--force`
appends regardless. For behavior and
output, see
[Adding Tests — From a graph you already have](../docs/adding-tests.md#from-a-graph-you-already-have).

### Grouped import for generated cases (opt-in)

Use these flags to put many generated graphs into a small number of sweeps.
A **sweep** is one graph template (`graph.template.json`) plus a list of cases
(`sweep.json`). The **skeleton hash** is the structural fingerprint of a graph
(node types, wiring, tensor set and tensor links such as
`ragged_offset_tensor_uid`); graphs with the same skeleton hash can share a
sweep. Without these flags, `import_graph.py` behaves as described
above.

```bash
python3 migration-scripts/import_graph.py \
    --graph exported/<case>/<case>.json \
    --bundle-dir integration-test-bundles/ --tier standard \
    --op-dir SdpaFwdPlan --topology <Topology> --case-id <id> \
    --template-attributes --template-alignment --skeleton-value-form \
    --only-op-dir --require-marked-sweep --strict
```

| Flag | Effect |
|---|---|
| `--op-dir NAME` | Operation folder for a new sweep. The default comes from the node types. |
| `--topology NAME` | Sweep folder for a new sweep. An append must go to the sweep in this folder, else the import fails. The tool never writes into a folder that holds another sweep. |
| `--template-attributes` | A new template replaces each non-object node attribute value with `${case.attributes.<key>}`. The key is the attribute path, joined with `__`. |
| `--template-alignment` | A new template replaces each tensor `alignment` with `${case.alignment.<uid>}`. The value comes from the top-level `alignment` object of the case values. |
| `--skeleton-value-form` | The skeleton hash also records the value form of each tensor: a baked `value`, its `value_type`, and `is_runtime_pass_by_value`. Different forms never share a sweep. A new template replaces each baked `value` with `${case.value}`. |
| `--only-op-dir` | Match only sweeps under `<bundle-dir>/<tier>/<op-dir>/`. A case never goes into another tier or another operation folder. |
| `--require-marked-sweep` | Match only sweeps whose `import_mode` equals the `import_mode` of this run. The `format` key is not compared. |
| `--superset-templates` | A graph that is the template of a sweep without some of its tensors joins that sweep (see below). |
| `--case-id ID` | Use this case id. The import fails if the sweep already has it. |
| `--sweep-format {pretty,compact}` | File layout of a new `sweep.json`. Grouped mode writes `compact` unless you give `--sweep-format pretty`. Use it only with another grouped-import flag. |

Matching includes the exact-duplicate check, so these filters apply to it too.

Each of these flags writes an `import_mode` object into a new `sweep.json`.
The object lists the template and skeleton flags in effect, for example
`{"grouping": ["skeleton-value-form", "template-alignment", "template-attributes"], "format": "compact"}`.
The `format` key records the file layout. `pretty` is `json.dump` with an
indent of 2. `compact` writes each member of a case, and each member of its
`values`, each on its own compact line. Each tensor of `values.tensors` gets its own
line. An append keeps the layout of the sweep and ignores `--sweep-format`.
A marker without `format` is `pretty`. Grouped mode writes `compact` unless
you give `--sweep-format pretty`.
A grouped template also replaces the top-level `is_override_shape_enabled` and
`min_required_engine_api_version` with `${case.<field>}`. hipDNN derives these
fields from the tensors, so they can differ between cases of one sweep.
Default mode does not append into sweeps that grouped mode wrote. Default
mode still reports `DUPLICATE` when a graph equals a case in such a sweep.
A run with only `--op-dir`, `--topology` or `--case-id` writes
`{"grouping": []}`, so default mode also never appends into that sweep.
`retier_bundles.py` keeps the `import_mode` object of a sweep and writes the
sweep in its layout. It refuses to put cases from sweeps with different
`grouping` values, or with different layouts, into one sweep.

#### Superset templates (`--superset-templates`)

With this flag, a graph with no sweep of its own skeleton can join a sweep
whose template holds every tensor of the graph, matched by tensor name, and
some more. The template without those tensors must have the graph's skeleton,
and the C++ loader must accept them as absent tensors (see [Optional tensors](../docs/file-formats.md#optional-tensors--absent_tensors)
in the file-formats document). A template in the `--tier` wins over one in
another tier; among templates in the same tier (or when none is), the
smallest one wins (ties: the sorted tensor names, then the path). With
`--topology`, only the sweep in that folder
can take the graph. When it cannot, the graph starts a new sweep there if the
folder is empty (the import prints a `NOTE` naming the other templates that
fit), and the import fails if the folder holds another sweep or a sweep in
another folder has the graph's own skeleton. The case gets:

- `absent_tensors`: the template uids of the tensors the graph lacks;
- the template's uids for its own tensors, in the graph and in the metadata
  `inputs`; the import fails if this moves an `inputs`, `aliases`,
  `start_offsets` or `live_rows` uid to another tensor, or names an absent tensor;
- `format_version: 2` in its metadata; the sweep gets `version: 2`.

The round-trip check prunes the absent tensors from the template first, then
expands the placeholders, then applies `tensor_patches`, as the C++ loader
does. A checked-in template is never rewritten: a new graph that holds the
template of another sweep starts its own sweep, and, without `--topology`, the
import prints `NOTE: missed merge` naming the other sweep. To get one sweep for a set of
graphs, import the graph with the most tensors first. When the chosen template
cannot reproduce the graph, the import fails; it does not try another template.

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

Add new tests as bundle cases with `import_graph.py`. C++ is not needed. You
do not need to write cases by hand. See
[Adding Tests](../docs/adding-tests.md#add-a-bundle).

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
| Ragged offset multiplier | `TensorAttributes` | sweep `values.tensors[uid].ragged_offset_multiplier` | Per-case, only on a tensor with a ragged offset |
| Node attrs | `ConvFpropAttributes` etc. | sweep `values.attributes.<key>` | Per-case, only if varies |
| Seed | `inputFillRecipes().setGlobalSeed()` | sweep `case.metadata.seed` | Per-case metadata |
| Distribution/range | `InputFillRecipes.fills()` | sweep `case.metadata.inputs.{uid}` | Per-case metadata |
| Per-op defaults | `FillInputs.cpp` | **NOT stored** | Re-derived from topology |
| Provenance | (new) | sweep `case.metadata.reference_source` | Per-case metadata |

## Re-tiering existing cases

To change which tier a case lives in (never what the case is), describe the
target placement in an assignment file and let `retier_bundles.py` apply it:

```bash
# Start from "everything stays where it is", then edit proposed_tier.
python3 retier_bundles.py --bundle-dir integration-test-bundles/ \
    --write-identity assignments.json

python3 retier_bundles.py --bundle-dir integration-test-bundles/ \
    --assignments assignments.json            # dry run: plan + checks, no writes
python3 retier_bundles.py --bundle-dir integration-test-bundles/ \
    --assignments assignments.json --apply
```

Every case must be assigned (an unassigned case would otherwise vanish) and two
cases with one id may not land in the same sweep. After `--apply` the tool
re-reads the tree and checks, per case, that the expanded graph, metadata,
`tensor_patches`, golden pointer and per-engine support claims are identical to
before; it exits non-zero if not. Topologies whose `graph.template.json` differs
between tiers are unified to the template with the most `${case.*}`
placeholders, and moved cases gain the matching explicit `attributes`.

Two things the tool cannot see: names that code refers to by tier
(`quick_<Op>_...` in gtest filters, TOML `filters`, `knownReferenceGaps`, the
`ffm-quick` ids in the provider YAMLs) and the `golden`/`dvc` data itself (only
the pointers move). Grep for the case ids you moved and re-run
`ctest -N -L <tier>` afterwards.

## Scripts

| Script | Purpose |
|---|---|
| `bundle_utils.py` | Shared utilities (skeleton hash, canonicalization, expansion, case-id) |
| `census.py` | List and classify all C++ test cases (the migration denominator) |
| `place_bundles.py` | Convert captured bundles into template+sweep format (Hop B) |
| `verify_migration.py` | Reconcile census ↔ capture ↔ sweep, byte-diff graphs + metadata (Hop C) |
| `import_graph.py` | Import a single graph with duplicate detection |
| `retier_bundles.py` | Move existing cases between tiers, proving every case is unchanged |
| `find_case.py` | Query cases by op, dtype, layout, shape, input range, or id |
| `diff_coverage.py` | Differential coverage: assert `pass_set_bundle ⊇ pass_set_cpp` (Hop D) |
| `run_capture_pipeline.sh` | Orchestrate all hops + verification layers |
| `test_migration.py` | Self-test on synthetic fixture (no binary needed) |

## How It Works

### Structure-hash grouping

Every graph gets a canonical skeleton fingerprint: node types in order,
tensor UID wiring (canonically renumbered), the tensor set, and the
tensor-level UID links (`ragged_offset_tensor_uid`). Graphs with the same
fingerprint share a topology. A UID remap changes the wiring, each tensor
`uid` and each tensor-level link together.

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
