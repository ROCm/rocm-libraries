# Integration Tests

Integration tests validate hipDNN provider plugins (engine libraries such as
`libmiopen_plugin.so` or `libhipblaslt_plugin.so`) by building a graph, running
it through the plugin's engine, and comparing the result against a reference.

The suite lives in `dnn-providers/integration-tests/` and builds one binary,
`hipdnn_integration_tests`. That binary is consumed by each provider
(miopen-provider, hipblaslt-provider, hip-kernel-provider, …) so that **one
graph test runs against every engine** — see
[Provider Integration](#provider-integration).

## Quick Start

```bash
# Standalone build — point to the plugin explicitly
./bin/hipdnn_integration_tests \
  --test-article /path/to/libmiopen_plugin.so

# Superbuild — plugin discovery is automatic
./bin/hipdnn_integration_tests

# Enable data-driven bundle/sweep tests (opt-in)
./bin/hipdnn_integration_tests --allow-bundles
```

## Two Ways to Test a Graph

There are two mechanisms for testing that a graph runs correctly on an engine.
**Pick the mechanism first — it decides everything else.**

| | **Bundles + sweeps** (default) | **C++ integration tests** (special cases) |
|---|---|---|
| What it is | Graph stored as JSON + a case matrix of shapes/dtypes/layouts | A `buildGraph()` function + `INSTANTIATE_TEST_SUITE_P` |
| Add a case | Edit JSON / run a tool — no compile | Write C++, recompile |
| Best at | "does this graph run and match a reference on this engine" | anything that is *not* just "run a graph and verify output" |
| Discovery | Auto-discovered from `integration_test_bundles/` | Registered in CMake per file |
| Runs against every engine | Yes | Yes |

> **Bundles and sweeps are now the default way to test that a hipDNN graph runs
> and verifies on an engine.** They decouple the test *data* (topology, shapes,
> dtypes, golden tensors) from the test *harness* (build, execute, verify), so a
> single template+sweep pair can replace dozens of near-identical C++ test
> registrations.
>
> **C++ integration tests are reserved for tests that exercise something *other
> than* "a graph runs with an engine"** — error/unhappy paths, API-contract
> behavior, serialization round-trips, benchmarking knobs, determinism, and
> pass-by-value semantics. See
> [C++ Integration Tests](#c-integration-tests-history--when-to-use).

## Bundle Formats: Single-Graph vs Template-Sweep

A *bundle* is a graph plus its pre-computed golden reference tensors. hipDNN
supports two bundle kinds (RFC 0011 §4.1). They differ only in whether one
graph JSON serves one case or many.

### Single-graph bundle (no sweep)

One graph, one set of golden tensors.

```
integration_test_bundles/{Tier}/{Operation}/{Layout}/{DataType}/{Name}/
    {Name}.json              # one concrete graph (committed to git)
    {Name}.tensors.dvc       # DVC pointer for this bundle's .bin tensors
    {Name}.tensor0.bin       # binary tensor data (DVC-tracked, in S3)
    ...
```

### Template-sweep bundle

One invariant topology (`graph.template.json`) with `${case.*}` placeholders,
plus a `sweep.json` case matrix that fills those placeholders per case. Each
expanded case gets its own golden pointer.

```
integration_test_bundles/{Tier}/{Operation}/{TopologyName}/
    graph.template.json      # topology skeleton with ${case.dims}, ${case.data_type}, ...
    sweep.json               # list of cases: values + golden path + metadata
    golden/{CaseId}/tensors.dvc
    golden/{CaseId}/tensor0.bin
    ...
```

`graph.template.json` holds the parts that never vary (node types, tensor
wiring/UIDs, `virtual` flags); `sweep.json` holds the parts that do (dims,
strides, dtypes, node attributes, seeds, input ranges). See
[`integration_test_bundles/README.md`](integration_test_bundles/README.md) for
the on-disk layout, DVC remote layout, and pull/push workflow, and
[`migration-scripts/README.md`](migration-scripts/README.md) for the exact
field mapping between the two.

### When to use which

**Default to a template-sweep bundle.** Use a straight single-graph bundle only
when a sweep would buy you nothing.

Use a **template-sweep** when:

- You are testing the *same topology* across several shapes, dtypes, or layouts
  (the common case — e.g. batchnorm inference over fp32/fp16 × nchw × small/large).
- You expect the case list to grow: adding a shape/dtype is one entry in
  `sweep.json`, not a new directory.
- You want the readable case-id filter surface (`{shape}_{dtype}_{layout}`) so
  `--gtest_filter` and `find_case.py` can slice the matrix.

Use a **straight single-graph bundle** (no sweep) when:

- There is exactly one concrete graph to test and no axis to vary over — a
  one-off regression graph, a specific customer/model layer, or a captured graph
  you want to pin byte-for-byte.
- The topology itself changes per case (different node counts/wiring), so cases
  cannot share one template. A sweep can only vary knob values, not structure;
  distinct structures are distinct bundles. (SDPA-forward is an example: each
  head-dim/mask/stats variant is generated as its own bundle rather than
  templatized.)
- Golden data comes from a bespoke per-case generator whose output does not map
  cleanly onto a single parameterized skeleton.

> Rule of thumb: **same skeleton, many knob values → sweep. One graph, or many
> skeletons → single-graph bundles.** When in doubt, `import_graph.py` decides
> for you: it groups by structure hash, appends to an existing sweep when the
> skeleton matches, and falls back to a standalone bundle when it does not.

## Adding a Bundle Test

New graph-runs-on-engine tests should be added as bundles — no C++ needed. The
recommended path is `import_graph.py`, which is dedup-aware and auto-assigns the
case id:

```bash
python3 migration-scripts/import_graph.py \
    --graph new_conv.json \
    --bundle-dir integration_test_bundles/
```

What happens:

1. It computes the graph's skeleton hash and finds matching topologies.
2. **Duplicate** (same graph + seed + inputs) → reports `DUPLICATE` and skips.
3. **New case for an existing topology** → appends to that `sweep.json`.
4. **New topology** → creates a new template+sweep directory.
5. The auto-generated case id is printed to stderr — that id is the gtest name.

Golden tensor data is tracked with DVC (stored in S3, not git). See
[`integration_test_bundles/README.md`](integration_test_bundles/README.md) for
adding/updating/removing the `.bin` data and the `dvc push`/`dvc pull`
workflow, and [`migration-scripts/README.md`](migration-scripts/README.md) for
the full tooling reference (`import_graph.py`, `find_case.py`, `place_bundles.py`,
the capture/verify pipeline, and manual `sweep.json` editing).

### Searching cases

```bash
# Find all batchnorm bundle cases
python3 migration-scripts/find_case.py --op Batchnorm

# Find cases that have an epsilon input
python3 migration-scripts/find_case.py --input epsilon

# Find cases where epsilon is in [-1,1]
python3 migration-scripts/find_case.py --input epsilon:-1,1

# Full detail for a hashed case id (includes the exact --gtest_filter)
python3 migration-scripts/find_case.py --id f446b9 --detail
```

### Verification modes

Bundle output can be verified against golden data or a live reference executor.
The mode is chosen with `--verification-mode` (or `HIPDNN_TEST_VERIFICATION_MODE`):

| Mode | Behavior |
|------|----------|
| `auto` (default) | golden → GPU ref → CPU ref → skip, in that order |
| `golden` | compare against DVC-fetched golden tensors only |
| `gpu` | compute the reference on the GPU ref executor |
| `cpu` | compute the reference on the CPU ref executor |

Golden data is optional: `--verification-mode gpu` (or `cpu`) runs the bundle
graphs without any DVC pull. Bundle registration itself is gated on
`--allow-bundles` (or `HIPDNN_TEST_ALLOW_BUNDLES=1`); without it, only the C++
tests run.

## Test Tiers

Tiers bound how long a run takes. They apply to both the C++ reference-executor
tests (via GTest prefixes) and to bundles (via the `{Tier}` path segment).

| Tier | GTest prefix | Bundle dir | CI cadence | Timeout |
|------|-------------|------------|------------|---------|
| Smoke | `Smoke` *(or no prefix)* — **catch-all** | `quick/` | Every commit / PR | 600s (10 min) |
| Standard | `Standard` | `standard/` | PR gate | 1800s (30 min) |
| Comprehensive | `Comprehensive` | `comprehensive/` | Nightly | 3600s (60 min) |
| Full | `Full` | `full/` | Weekly | 7200s (120 min) |

Timeouts can be overridden per binary via `SMOKE_TIMEOUT`, `STANDARD_TIMEOUT`,
`COMPREHENSIVE_TIMEOUT`, and `FULL_TIMEOUT` arguments to
`add_tiered_test_target()`.

### Smoke is a catch-all

The smoke ctest entry uses an exclusion filter
(`-Standard*:Comprehensive*:Full*`). Every test that does **not** start with
`Standard`, `Comprehensive`, or `Full` runs in smoke automatically:

```cpp
// Runs in smoke — has Smoke prefix
INSTANTIATE_TEST_SUITE_P(Smoke, MyFixture, ...);

// Also runs in smoke — no tier prefix, caught by the exclusion filter
TEST(MyFeature, BasicBehavior) { ... }
TEST_F(MyFixture, EdgeCase) { ... }
```

If smoke starts timing out, a large shape is missing its tier prefix.

### How tiers cascade

Each higher ctest label includes all lower tiers:

```
ctest -L quick           →  [smoke]
ctest -L standard        →  [smoke + standard]
ctest -L comprehensive   →  [smoke + standard + comprehensive]
ctest -L full            →  [smoke + standard + comprehensive + full]
```

> **Note:** The ctest label uses `quick` for the smoke tier
> (backlog: rename to `smoke` for consistency).

## Running Tests

| Method | Command | Use case |
|--------|---------|----------|
| ctest | `ctest -L quick` | CI and local tier runs |
| ninja | `ninja unit-check` / `ninja check` | Local shortcut (smoke / all) |
| Direct | `./bin/hipdnn_gpu_ref_tests --gtest_filter="Smoke*"` | Debugging a specific test |

Bundle suites register under gtest as `{tier}_{Op}_{Topology}` (e.g.
`quick_Batchnorm_Default`), with each case named by its case id:

```bash
# Run all quick-tier bundles
--gtest_filter='quick_*'

# Run all batchnorm cases (any tier)
--gtest_filter='*Batchnorm*'

# Run one exact case (by hash suffix)
--gtest_filter='*f446b9*'
```

> **GTest filter syntax:** `-Standard*:Comprehensive*:Full*` uses a single
> leading dash. In GTest, only the first `-` starts the negative section.
> Using `:-` between patterns does **not** negate — the dash becomes literal.

## Provider Integration

Each provider consumes `hipdnn_integration_tests` as a CMake package and runs it
against its own plugin. This is how the shared graph suite validates every
engine (miopen, hipblaslt, hip-kernel) from one place.

### How a provider wires it in

The suite installs a CMake package (`hipdnn_integration_tests`) that exports the
test binary target plus the helper module
[`cmake/HipdnnIntegrationTestHelpers.cmake`](cmake/HipdnnIntegrationTestHelpers.cmake).
A provider's `CMakeLists.txt` finds the package and registers an
`<provider>-external-integration-check` target:

```cmake
# Prefer the superbuild target; fall back to the installed package (standalone).
if(NOT TARGET hipdnn_integration_tests)
    find_package(hipdnn_integration_tests CONFIG QUIET)
endif()

if(TARGET hipdnn_integration_tests)
    add_external_integration_test_target(
        TARGET_NAME    ${PROJECT_NAME}-external-integration-check
        PLUGIN_TARGET  miopen_plugin              # the provider's plugin .so target
        ENGINE_NAME    MIOPEN_ENGINE              # passed via --test-engine
        INSTALL_SUBDIR miopen_plugin
        TEST_CONFIG    ${CMAKE_CURRENT_SOURCE_DIR}/config/MIOPEN_ENGINE.toml
        TEST_CATEGORIES_YAML ${MIOPENPROVIDER_INTEGRATION_CATEGORIES_YAML}
    )
endif()
```

`add_external_integration_test_target()` invokes the shared binary as:

```
hipdnn_integration_tests --test-article <plugin.so> --test-engine <ENGINE> [--test-config <toml>] [--gtest_filter=...]
```

- `--test-article` resolves to the provider's plugin `.so` at build time
  (`$<TARGET_FILE:...>`).
- `--test-engine` pins the run to that provider's engine, so unsupported ops
  `SKIP` rather than fall through to another loaded engine.
- `--test-config` supplies per-test tolerance overrides (a TOML file the
  provider owns, e.g. `config/MIOPEN_ENGINE.toml`).
- `TEST_CATEGORIES_YAML` generates tier-labelled ctest suites so
  `ctest -L quick|standard|...` selects tiers for the external run too.

Both the superbuild (target already present) and standalone provider builds
(`find_package`) are supported; if the package is not found the target is
skipped with a status message.

### Per-provider category filtering

The shared binary uses hipDNN GTest naming, which differs from the provider's
own `*_plugin_tests`. Each provider therefore keeps a separate
`test_categories_integration.yaml` that is applied *only* to the external
`hipdnn_integration_tests` run (the provider's own `test_categories.yaml`
covers its native binaries). A provider whose engine only implements a subset of
ops can leave every tier at the `*` pattern — the harness's generic
engine-support check (`checkEngineSupportOrSkip` / `verifyGraph`) skips ops the
engine does not support.

### Running the cross-provider suite

```bash
# Build + run the external integration suite for a provider
cmake --build build --target miopen-provider-external-integration-check

# Or by tier via ctest, from the provider build/install dir
ctest -L quick
```

## C++ Integration Tests (History & When to Use)

Before bundles, every integration test was C++: a `buildGraph()` function plus
an `INSTANTIATE_TEST_SUITE_P` with hardcoded shape/dtype/layout lists. Each new
shape meant new C++ and a recompile, and the graphs were hard to audit or reuse
across engines.

`ALMIOPEN-2221` / `ALMIOPEN-2279` introduced the bundle+sweep format and a
migration pipeline (`migration-scripts/`) that captured the existing C++ graph
tests to JSON, grouped them by structure into template+sweep bundles, and
proved — byte-for-byte and behaviorally — that turning the C++ graph tests off
lost no coverage. As a result, **"does this graph run and verify on an engine"
is now a bundle concern, not a C++ one.**

C++ integration tests still exist and still matter — but only for what bundles
cannot express. The remaining C++ tests live in two places:

- `src/integration-tests/{op}/` — shared cross-provider C++ tests built into
  `hipdnn_integration_tests` (conv, matmul, sdpa, batchnorm, layernorm,
  rmsnorm, reduction, pointwise).
- `<provider>/integration_tests/` — provider-local C++ tests (e.g.
  `miopen_plugin_integration_tests`) for behavior specific to one plugin.

Write a **C++ integration test** (not a bundle) when the test exercises
something other than a graph running and matching a reference:

- **Unhappy / error paths** — unsupported dtypes, invalid layouts, unsupported
  activation combinations (e.g. `IntegrationGpuBatchnormUnsupportedDataTypes`,
  `IntegrationGpuBatchnormUnhappyLayouts`).
- **API-contract behavior** — pass-by-value scalar semantics, `is_supported`
  queries, benchmarking knobs (`IntegrationGpuPassByValue`,
  `IntegrationGpuBenchmarkingKnob`, `IntegrationIsSupportedExtPerformance`).
- **Serialization round-trips** — graph serialize/deserialize identity
  (`IntegrationConvForwardSerializeRoundTrip`).
- **Determinism / repeated-run invariants** (`IntegrationGpuDeterministic`).

If a proposed C++ test is really just "build graph X, run it, compare to a
reference," it belongs in a bundle instead. Use `--capture-bundles` and the
migration scripts to convert an existing C++ graph test into a bundle rather
than adding another parameterized instantiation.

## Adding a New Reference-Executor Operation

The `tests/` tree holds the GPU/CPU *reference executor* tests (built as
`hipdnn_gpu_ref_tests`) that validate the reference itself — separate from the
provider-facing graph tests above. To add a new op there:

### Directory layout

```
tests/
  gpu-ref/
    ConvShapeCase.hpp              # Shape struct + byTag()
    ConvShapeCatalog.hpp           # getSmall/getMedium/getLargeEdge/getLargeStress
    TestGpuFpReferenceConvolution.cpp
  my_new_op/
    MyNewOpShapeCase.hpp
    MyNewOpShapeCatalog.hpp
    TestMyNewOp.cpp
```

### Step 1 — CMake registration

Register the test binary in `tests/CMakeLists.txt`:

```cmake
add_tiered_test_target(hipdnn_my_new_op_tests ${CMAKE_CURRENT_BINARY_DIR})
```

### Step 2 — Shape catalog

Create a shape catalog following the tier pattern in
[`tests/gpu-ref/ConvShapeCatalog.hpp`](tests/gpu-ref/ConvShapeCatalog.hpp).

### Step 3 — C++ test tiers

New parameterized test suites **must** define all four tiers:

```cpp
INSTANTIATE_TEST_SUITE_P(Smoke,         MyNewOp2dTestFp32, ::testing::ValuesIn(getSmallCases()),     byTag());
INSTANTIATE_TEST_SUITE_P(Standard,      MyNewOp2dTestFp32, ::testing::ValuesIn(getMediumCases()),    byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive, MyNewOp2dTestFp32, ::testing::ValuesIn(getLargeEdgeCases()), byTag());
INSTANTIATE_TEST_SUITE_P(Full,          MyNewOp2dTestFp32, ::testing::ValuesIn(getLargeStressCases()), byTag());
```

`byTag()` uses the shape's `tag` field as the test name so failures show
`Smoke/MyOp2dTestFp32.Runs/n8c64k32_f3x3_s1_p1` instead of `.../7`.

### Adding a new convolution shape

Add to the appropriate function in
[`tests/gpu-ref/ConvShapeCatalog.hpp`](tests/gpu-ref/ConvShapeCatalog.hpp).
Existing `INSTANTIATE_TEST_SUITE_P` calls pick up new shapes automatically.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Engine 'X' is not loaded` | Pass `--test-article /path/to/plugin.so`, or run from a superbuild |
| Bundle tests don't run | Pass `--allow-bundles` (or set `HIPDNN_TEST_ALLOW_BUNDLES=1`) |
| Tests can't find bundle data | `dvc pull` the op, or run with `--verification-mode gpu` to skip golden comparison |
| Smoke tier timing out | A shape is missing its tier prefix — check `INSTANTIATE_TEST_SUITE_P` prefixes |
| `No tests matched the filter` | Use a single `-` for negative filters: `-Standard*:Comprehensive*:Full*` |

## See Also

- [`integration_test_bundles/README.md`](integration_test_bundles/README.md) —
  on-disk bundle layout, DVC remotes, and add/update/remove/pull/push workflow.
- [`migration-scripts/README.md`](migration-scripts/README.md) — capture →
  place → verify pipeline, `import_graph.py`, `find_case.py`, and the C++ → bundle
  field mapping.
- [RFC 0011 — Golden Reference Validation](../../projects/hipdnn/docs/rfcs/0011_GoldenReferenceValidation.md)
  — the bundle/sweep naming spec (§4.1) and design rationale.
