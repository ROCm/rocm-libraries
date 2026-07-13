# Migration Scripts — C++ Graph Tests to Bundle Format

These scripts convert C++ integration graph tests into the compressed
template+sweep bundle format (ALMIOPEN-2221).

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
```

## How to Run

### Prerequisites

Build the integration test binary (no GPU required):

```bash
cmake --build build --target integration_tests
```

### Step 1: Census (optional) — see what exists

```bash
python3 migration_scripts/census.py build/bin/integration_tests
```

Runs `--gtest_list_tests` and classifies every test case as `graph`
(C++ graph test to migrate), `bundle` (already a bundle), or `other`.
Outputs a JSON manifest — this is the denominator for tracking migration
progress.

### Step 2: Capture (Hop A) — serialize C++ graphs as JSON bundles

```bash
./build/bin/integration_tests --capture-bundles captured_bundles
```

Each C++ graph test runs its `buildGraph()`, then the capture hook
serializes the resulting graph JSON and metadata into:

```
captured_bundles/{SuiteName}/{CaseName}/{CaseName}.json
captured_bundles/{SuiteName}/{CaseName}/{CaseName}.meta.json
```

This works on CPU — no GPU needed. The capture hook fires inside
`verifyGraph()` before any GPU execution.

### Step 3: Place (Hop B) — compress into template+sweep

```bash
python3 migration_scripts/place_bundles.py \
    --capture-dir captured_bundles \
    --output-dir dnn-providers/integration-tests/integration_test_bundles
```

Groups captured graphs by **structure** (node types + wiring + tensor
set), not by suite name. Graphs that share the same topology but differ
in shapes/dtypes/layouts/attributes collapse into one template+sweep.

Output layout:

```
integration_test_bundles/{tier}/{Operation}/{TopologyName}/
    graph.template.json    # graph with ${case.*} placeholders
    sweep.json             # array of cases with concrete values
```

Flags:
- `--dry-run` — report what would be written without writing
- `--no-verify` — skip round-trip verification (not recommended)

The script also writes `topology_map.json` into `.migration_reports/`
for human review of auto-generated topology names.

## How It Works

### Structure-hash grouping

Every graph gets a canonical skeleton fingerprint: node types in order,
tensor UID wiring (canonically renumbered), and the tensor set (which
UIDs exist, virtual flags). Graphs with the same fingerprint share a
topology — they become one template with per-case values.

### Derive, don't classify

Instead of maintaining a per-op allowlist of which fields are
"structural" vs "knobs," the script uses a key-name rule:

- **Structural** (fixed in template): `node.type`, `*_tensor_uid` keys,
  `tensor.uid`, `tensor.virtual`, tensor set
- **Knob** (templatized if varies): everything else — dims, strides,
  dtypes, node attributes, inline tensor values

This is safe because of the **safety asymmetry**: treating a knob as
structural just means more topologies (under-compressed but correct),
while treating something structural as a knob would cause a verify
failure (caught, falls back to standalone).

### Verify gate

Every expanded case is compared byte-for-byte against the original
captured graph. Any mismatch falls back to a standalone bundle. Nothing
is silently dropped.

## Scripts

| Script | Purpose |
|---|---|
| `census.py` | List and classify all C++ test cases (the migration denominator) |
| `place_bundles.py` | Convert captured bundles into template+sweep format |
