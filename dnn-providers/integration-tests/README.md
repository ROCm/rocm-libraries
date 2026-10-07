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

# Bundle/sweep tests run by default. Turn them off to run only the C++ tests
# compiled into the binary — see "Bundles are the CI driver" below
./bin/hipdnn_integration_tests --no-bundles
```

## Two Ways to Test a Graph

There are two mechanisms for testing that a graph runs correctly on an engine.
**Pick the mechanism first — it decides everything else.**

| | **Bundles + sweeps** (default) | **C++ integration tests** (special cases) |
|---|---|---|
| What it is | Graph stored as JSON + a case matrix of shapes/dtypes/layouts | A `buildGraph()` function + `INSTANTIATE_TEST_SUITE_P` |
| Add a case | Edit JSON / run a tool — no compile | Write C++, recompile |
| Best at | "does this graph run and match a reference on this engine" | anything that is *not* just "run a graph and verify output" |
| Discovery | Auto-discovered from `integration-test-bundles/` | Registered in CMake per file |
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

### Bundles are the CI driver

"Default" is true of both **authoring** and **execution**:

- **Execution.** Bundle registration is on by default. `--no-bundles` (or
  `HIPDNN_TEST_ALLOW_BUNDLES=0`) is the opt-out, leaving only the C++ tests
  compiled into the binary.
- **Build.** The C++ graph tests under `src/integration-tests/` no longer build
  by default: they are gated behind `-DBUILD_CPP_GRAPH_TESTS=ON`, which is
  `OFF` unless a developer opts in on their branch. So the provider CI checks
  run bundles and nothing else.
- **Authoring.** New graph-verification coverage must be a bundle. CMake
  enforces this — see [Adding a C++ test](#adding-a-c-test-the-cmake-rule).

## Bundle Formats: Single-Graph vs Template-Sweep

A *bundle* is a graph, optionally paired with pre-computed golden reference
tensors. hipDNN supports two bundle kinds (RFC 0011 §4.1), and either kind can
be **full** (tensor data included) or **graph-only** (no tensor data — the
engine's output is compared against a live GPU/CPU reference executor
instead). The two kinds below differ only in whether one graph JSON serves
one case or many; golden data is optional in both.

### Single-graph bundle (no sweep)

One graph, optionally one set of golden tensors:

```
integration-test-bundles/{Tier}/{Operation}/{Layout}/{DataType}/{Name}/
    {Name}.json              # one concrete graph (committed to git)
    {Name}.tensors.dvc       # optional — omit for a graph-only bundle
    {Name}.tensor0.bin       # optional — DVC-tracked, in S3
    ...
```

### Template-sweep bundle

One invariant topology (`graph.template.json`) with `${case.*}` placeholders,
plus a `sweep.json` case matrix that fills those placeholders per case. A
case's `golden` pointer is optional — a sweep case with no `golden` entry (or
no `.dvc`/`.bin` fetched) is a graph-only case, verified against the GPU/CPU
reference executor instead of golden comparison.

```
integration-test-bundles/{Tier}/{Operation}/{TopologyName}/
    graph.template.json      # topology skeleton with ${case.dims}, ${case.data_type}, ...
    sweep.json               # list of cases: values + optional golden path + metadata
    golden/{CaseId}/tensors.dvc   # optional per case
    golden/{CaseId}/tensor0.bin   # optional per case
    ...
```

`graph.template.json` holds the parts that never vary (node types, tensor
wiring/UIDs, `virtual` flags); `sweep.json` holds the parts that do (dims,
strides, dtypes, node attributes, seeds, input ranges). See
[`integration-test-bundles/README.md`](integration-test-bundles/README.md) for
the on-disk layout, DVC remote layout, and pull/push workflow, and
[`migration-scripts/README.md`](migration-scripts/README.md) for the exact
field mapping between the two.

To run against part of the tree, point `--golden-data-dir` (`--gd`) at a directory
that holds only the pieces you want. They can be copies or directory symlinks into
an existing tree, e.g. `quick/SdpaFwd -> <install>/lib/integration-test-bundles/quick/SdpaFwd`.
Discovery follows directory links at any depth. Tests take their names from the
path through the link, so a linked subtree gets the same test names as a copy. A
link that would lead the walk back into a directory it is already inside (one of
the link's own parents, including the parents of `--gd` itself) is skipped with a
warning, and so is a directory the run is not allowed to list.

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
    --bundle-dir integration-test-bundles/
```

What happens:

1. It computes the graph's skeleton hash and finds matching topologies.
2. **Duplicate** (same graph + seed + inputs) → reports `DUPLICATE` and skips.
3. **New case for an existing topology** → appends to that `sweep.json`.
4. **New topology** → creates a new template+sweep directory.
5. The auto-generated case id is printed to stderr — that id is the gtest name.

### What is a "sweep", and how do you author one?

A **sweep** is the `sweep.json` case matrix belonging to a template-sweep
bundle (see [Bundle Formats](#bundle-formats-single-graph-vs-template-sweep)):
one row per case, each with concrete `values` for the template's `${case.*}`
placeholders plus per-case metadata and an optional `golden` pointer.

**Never hand-write a `sweep.json` from scratch.** Existing sweeps (some are
thousands of lines, e.g. `quick/Batchnorm/Default/sweep.json`) are tool
output, not hand-authored — `import_graph.py` appends one case at a time and
auto-assigns its id; nobody typed those rows by hand.

**Gap: there is no tool to author a brand-new sweep's full case matrix from
scratch.** `import_graph.py` only appends cases you already have as concrete
graph JSON, one call per case; there's no "generate N cases for these
dims/dtypes/layouts" command for a topology that doesn't already exist
anywhere. Today's mitigation is a round trip through C++: write the case
matrix as a normal C++ `INSTANTIATE_TEST_SUITE_P` graph test (fast to iterate
over many shapes/dtypes in one parameterized fixture), then run the existing
export tooling — `--capture-bundles` (Hop A) to serialize each parameterized
case as JSON, then `place_bundles.py` (Hop B, bulk) or `import_graph.py`
(Step 5, incremental) to fold them into a `sweep.json` — and delete the C++
test once the exported bundle is verified equivalent (see
[`migration-scripts/README.md`](migration-scripts/README.md)). A direct
sweep-authoring tool that skips the C++ detour is a known future need, not
yet built.

Golden tensor data is tracked with DVC (stored in S3, not git). See
[`integration-test-bundles/README.md`](integration-test-bundles/README.md) for
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

Bundle output is verified against golden data or a live reference executor. The mode
is chosen with `--verification-mode` (or `HIPDNN_TEST_VERIFICATION_MODE`):

| Mode | Behavior |
|------|----------|
| `auto` (default) | golden → GPU ref → CPU ref → skip, in that order |
| `golden` | compare against DVC-fetched golden tensors only; **FAIL if a bundle has none** |
| `gpu` | compute the reference on the GPU ref executor |
| `cpu` | compute the reference on the CPU ref executor |

`auto` is the mode with a fallback chain. An explicit mode is a demand for a
specific oracle, so `golden` on a bundle with no golden data is a failure, not a
skip — `dvc pull` the op, or use `auto`.

Each verification test body prints the oracle that graded it, between its
`[ RUN ]` and result lines, and the coverage summary totals them:

```
[ VERIFIER ] gpu_ref: .../quick/SdpaFwd/bshd/fp16/hd64_nomask_mqa/Small.json
...
Verified by: golden 0, gpu_ref 9, cpu_ref 0, none 3
```

`golden`, `gpu_ref` and `cpu_ref` name what the outputs were compared against,
whether the comparison passed or failed. `none` means nothing was compared: a skip
(the engine declined, or no reference could run the op), a failure before the
comparison, or a bundle whose `enforcement_level` stops short of comparing. With
`--gtest_repeat=N` the line counts the last iteration, like the Passed, Skipped and
Failed counts above it.

Golden data is optional in the other modes: `--verification-mode gpu` (or `cpu`)
runs the bundle graphs without any DVC pull. Bundle registration is on by default;
pass `--no-bundles` (or `HIPDNN_TEST_ALLOW_BUNDLES=0`) to leave only the C++ tests
that were compiled into the binary.

By default the comparison runs where the expected values live, so the reference you
select is also the validator you get. Output checked against the GPU reference is
compared on the device (`GpuFpReferenceValidation`, or `GpuFpReferenceRmsValidation`
for a `[[validator_overrides]]` RMS check); output checked against the CPU reference
or against golden data is compared on the host (`CpuFpReferenceValidation` /
`CpuFpReferenceMiopenRmsValidation`). That holds for every mode, including each step
of the `auto` fallback chain, and for C++ graph tests under
`--reference-executor gpu|cpu`. A `[[validator_overrides]]`
`"allclose_matching_infinities"` check has a host validator only: a comparison
that resolves to the device fails that tensor instead of running it.

`--validator auto|cpu|gpu` (or `HIPDNN_TEST_VALIDATOR`) overrides that choice for
every comparison in the run, independently of the reference. `auto` (the default)
follows the reference as above; `cpu` always uses the host validators; `gpu` always
uses the device validators, copying golden data or a CPU reference's output to the
device first — so it needs a GPU even where the reference does not. An explicit
`--validator auto` wins over the env var like any other value. Only the pass/fail
call moves: a failure report is always built on the host from the read-back values.

### Validating golden data itself

The `hipdnn_golden_data_tests` binary runs a **separate suite** that recomputes each
bundle's outputs with a reference executor and compares them against the checked-in
golden `.bin` data. No engine is loaded and no support claims are involved — it
validates our data, not a provider. Suites are named `…_CpuRef` / `…_GpuRef`.

It validates against both references by default; `--reference cpu|gpu|both` narrows
that. The CPU reference is host-only and needs no GPU; the GPU one skips without a
device. Comparisons run on the host, where the golden data is loaded, unless
`--validator gpu` moves them to the device — which then makes the CPU suite need a
device too.

A registered test has no skip path in its body: a test is registered only when the
bundle has golden data and every node type in its graph is in that reference's
required-op set, so a reference that cannot run the graph is a failure. Bundles
outside a lane's set are absent from that lane, and the counts — plus the ops
responsible — are printed at registration. With both lanes selected, a golden
bundle that neither lane registered a test for fails as `<bundle>_Unvalidated`.

Golden `.bin` blobs are DVC-managed, so a tree that has not run `dvc pull` in
`integration-test-bundles/` registers nothing and says so.

This replaces the former `--verification-mode=golden-check`, and the
`--validate-golden-data` flag that briefly stood in for it: golden-data validation
is its own binary, not a mode of the engine harness.

### Support claims

A bundle may carry a `.support.json` sidecar promising that a named engine supports
that graph on a given arch and platform. `--enforce-support-claims` (which requires
`--test-engine`) turns a broken promise into a test failure instead of a silent
skip, and prints a JSON summary of every verdict. Claims are checked for the single
engine under test. Enforcement is on by default, so the CTest registrations and a
run by hand behave alike; pass `--enforce-support-claims=false` to keep the summary
but let a broken claim stay green. See
[`docs/support-claim-enforcement.md`](docs/support-claim-enforcement.md).

## Test Tiers

Tiers bound how long a run takes. They apply to both the C++ reference-executor
tests (via GTest prefixes) and to bundles (via the `{Tier}` path segment).

| Tier | GTest prefix | Bundle dir | CI cadence | Timeout |
|------|-------------|------------|------------|---------|
| Smoke | `Smoke` *(or no prefix)* — **catch-all** | `quick/` | Every commit / PR | 600s (10 min) |
| Standard | `Standard` | `standard/` | PR gate | 1800s (30 min) |
| Comprehensive | `Comprehensive` | `comprehensive/` | Nightly | 3600s (60 min) |
| Full | `Full` | `full/` | Weekly | 7200s (120 min) |

Timeouts are configured per tier via `category_timeouts` in
[`test_categories.yaml`](test_categories.yaml).

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

## Knob catalog sweep

`--sweep-knob-catalog` is an offline mode of `hipdnn_integration_tests` that validates
**every configuration an engine can serve a graph with**, not just its default.
A normal bundle run builds one plan per graph (`HEURISTICS_CHOICE`) and executes
only the top-ranked configuration, and `--test-engine` pins the engine, not its
configuration. A kernel-ingestor engine can match tens or hundreds of kernels to one
graph; only the first is ever checked. The sweep asks each engine for its knob
*catalog* (`Graph::get_knob_catalog_for_engine`), creates a plan for every entry with
`create_execution_plan_ext`, runs it, and compares it to the reference.

It is a separate mode, not a flavour of the verification run, and it is **not part of
CI**. Runtime-compilation engines compile every entry, so a full sweep can take hours.

### Running it

```bash
export HIPDNN_CACHE_DIR=$(mktemp -d)        # must be absent or empty
unset HIPDNN_FORCE_BENCHMARKING             # must be unset (any value, even 0, refuses)

./bin/hipdnn_integration_tests \
  --sweep-knob-catalog \
  --catalog-report sweep.jsonl \
  --test-engine hipkernel:Pointwise \
  --verification-mode gpu \
  --gtest_filter='quick_Pointwise*'
```

The graphs are the bundle graphs, discovered and loaded exactly as in a verification
run, and `--gtest_filter`, `--test-engine`, `--golden-data-dir`, `--test-article`,
`--test-config` and `--verification-mode` mean what they mean there. In sweep mode
`--gtest_filter` selects graphs by their bundle name (`Suite.Test`); the sweep's own
tests are the only ones that run (the C++ graph tests are not).
`--test-engine X` sweeps only engine `X`; without it every engine the graph lists
is swept. Tolerances are the graph's own (the same atol/rtol, and the same
`[[tolerance_overrides]]` / `[[validator_overrides]]` from `--test-config`, a normal run
of that bundle uses).

| Flag | Meaning |
|---|---|
| `--sweep-knob-catalog` | Enable the mode. |
| `--catalog-report <path>` | **Required.** JSON Lines report. Opened for append (never truncated) and flushed after every record, so an interrupted run keeps what it finished. |
| `--catalog-filter <expr>` | Run only the matching (graph, engine) pairs; everything else is still enumerated and recorded `FILTERED`. |
| `--catalog-shard <i>/<n>` | Sweep only the graphs shard `i` of `n` owns (0-based, `i < n`). |

`--catalog-*` flags without `--sweep-knob-catalog` are refused, not ignored. A sweep
also cannot be combined with `--write-support-claims` or `--capture-bundles`, and needs a
HIP device (the no-GPU "skip with exit 0" shortcut of the other modes does not apply:
a sweep that sweeps nothing must not look green).

### Refusals at startup

The sweep exits with status 1, before loading any plugin, when:

- `--catalog-report` is missing, or its file cannot be opened for append;
- `HIPDNN_FORCE_BENCHMARKING` is set **to anything**. It overrides `global.benchmarking`
  for every plan, so the sweep would not be building the configurations the catalog
  lists. An empty value counts as set; `0` counts as set;
- `HIPDNN_CACHE_DIR` is unset or empty, or does not name a **fresh** directory. Fresh
  means: it does not exist yet, or it exists, is a directory, and is empty. hipDNN
  persists engine rankings under it, and a ranking cached by an earlier run would change
  which kernels a later run is offered. (An empty `HIPDNN_CACHE_DIR` is treated by hipDNN
  itself as unset, so it is refused rather than allowed to fall back to the per-user cache.)

### Filter expressions

```
expr   := clause (',' clause)*      all clauses must hold
clause := field op values
field  := graph | engine            the bundle's Suite.Test name / the engine name
op     := =  !=  ~  !~              glob / not-glob / substring / not-substring
values := pattern ('|' pattern)*    any alternative may match
```

Globs match the whole name, case-sensitively; `*` is any run of characters, `?` exactly
one. Whitespace around tokens is ignored; a pattern cannot contain `,` or `|`.

```
--catalog-filter 'engine=hipkernel:*'
--catalog-filter 'graph~Pointwise,engine!=MIOPEN_ENGINE'
--catalog-filter 'engine=MIOPEN_ENGINE|HIP_MLOPS_ENGINE,graph!~Bf16'
```

The filter narrows the work, never the accounting: a filtered-out pair still has its
catalog fetched, and each of its entries is written with status `FILTERED` (a graph the
filter excludes outright is not executed at all, and needs no reference).

### Sharding

`--catalog-shard i/n` splits a sweep across processes or hosts. The shard unit is the
**graph**, not the entry: partitioning by entry would make every shard build the graph,
fetch the catalog and run the reference, whereas a graph-level shard does that work once
for the graphs it owns. A graph belongs to shard `fnv1a64(Suite.Test) mod n` --
a function of the name alone, so shards agree across hosts and do not depend on discovery
order or on `--gtest_filter`. Every graph belongs to exactly one shard; run all `n` and
concatenate the reports. Give each shard its own `HIPDNN_CACHE_DIR` and its own report
file, so concurrent shards never share a cache or interleave writes.

Shard by graph is coarse when a few graphs dominate (one graph with hundreds of
runtime-compiled entries is one shard's whole job). Use `--catalog-filter` with `graph`
clauses to give such a graph a process of its own.

### What happens per graph

1. The graph is built **once** (`from_binary`) and that same `Graph` object is used for
   everything below.
2. `get_engine_configs` lists the engines (only `--test-engine`'s if given). A graph no
   engine supports, or the pinned engine does not list, is skipped here, before any
   reference work.
3. Only if at least one listed engine will actually be swept (with `--catalog-filter`:
   has a (graph, engine) pair the filter lets through): the inputs are filled once from
   the bundle's seeded recipes and the reference (golden data, GPU or CPU reference, per
   `--verification-mode`) runs **once**; its outputs are copied to the host once and are
   never handed to a plan. A reference problem therefore only ever fails a graph that an
   engine serves.
4. Per engine, `get_knob_catalog_for_engine` returns the entries. If it fails, or
   returns nothing, the engine gets one `NO_CATALOG` record and the graph fails.
5. Per entry: `create_execution_plan_ext(engineId, entry)`, `check_support`,
   `build_plans`, workspace allocation, a **refill of the inputs**, **fresh outputs
   prefilled with NaN**, execute, and a comparison of the graph's registered outputs
   against the reference. The comparison builds its validators afresh each time.

Because outputs are reset to NaN before every entry, a kernel that writes nothing is
reported `FAIL_NUMERIC` (it left the sentinel) instead of passing on the previous
entry's answer.

A graph is **skipped** (not failed) where a normal run would skip it: a device/VRAM/arch
guard or `[[test_skips]]` entry applies, no engine supports the graph (the frontend's
`GRAPH_NOT_SUPPORTED` from `get_engine_configs`; message `no engine supports this graph`),
the pinned engine does not list it, or no reference can run it (including bundles whose `enforcement_level` is below `FULL`, whose
outputs are not compared). A skip is still written to the report as a reconcile record.
`--fail-on-unsupported` turns the two "no engine / pinned engine does not list the graph"
skips into failures, as in the verification run. A graph whose reference *breaks*, that
cannot be built, or whose engine listing fails with any other error, fails.

### Statuses

| Status | Meaning |
|---|---|
| `PASS` | built, executed, every output matches the reference |
| `FAIL_NUMERIC` | executed; an output is out of tolerance, or holds a NaN (even one the validator's tolerance would let through) |
| `FAIL_BUILD` | `create_execution_plan_ext`, `check_support` or `build_plans` failed. This includes the plugin refusing the entry with `INVALID_VALUE` (an unsatisfiable knob filter) and an engine that cannot run the graph on this device: **never skipped** |
| `FAIL_EXECUTE` | the plan built but workspace, input preparation or execution failed (or the entry threw) |
| `FILTERED` | excluded by `--catalog-filter`; enumerated and recorded, not run |
| `NO_CATALOG` | the engine returned no catalog for this graph |

No entry an engine returns is ever skipped: each is run and classified, or `FILTERED`.
The process exits non-zero if any graph failed (any `FAIL_*`, any `NO_CATALOG`, a graph
error), or if the sweep enumerated no entries at all.

### Report (JSON Lines)

One **entry record** per catalog entry:

```json
{"graph":"quick_Pointwise_nchw_fp32_Small.Small","engine_id":123,"engine_name":"hipkernel:Pointwise",
 "entry_index":3,"knobs":[{"knob_id":"block_size","value":64}],
 "status":"FAIL_NUMERIC","max_abs_err":4.5,"mismatch_pct":100.0,
 "message":"output y mismatches the reference (max_abs_err=4.5, mismatched=64/64, nan=64)",
 "compile_time_ms":812.4,"device_arch":"gfx1151","hipdnn_version":"1.2.3.4"}
```

| Field | Notes |
|---|---|
| `graph` | the bundle's `Suite.Test` name |
| `engine_id`, `engine_name` | |
| `entry_index` | position in the engine's catalog; **informational** -- catalogs depend on graph, device and shipped kernel packs, so the knob set identifies an entry. `-1` on the record of an engine with no catalog |
| `knobs` | array of `{knob_id, value}`; values keep their type (the integer `2` is not the string `"2"`) |
| `status` | see above |
| `max_abs_err`, `mismatch_pct` | worst absolute error over all outputs, and the percentage of elements out of tolerance or NaN. `null` when not measured (not run, or no element-wise measure for the output type; `mismatch_pct` is also `null` for RMS-graded outputs) |
| `message` | failure detail; empty on `PASS` |
| `compile_time_ms` | time in `create_execution_plan_ext` + `check_support` + `build_plans`: where runtime-compilation engines spend their time. `null` when the entry was not built |
| `device_arch`, `hipdnn_version` | |

And one **reconcile record** per graph, told apart by `"kind":"reconcile"`, written
after that graph's entries (a skipped or failed graph gets one too, with `entries` 0):

```json
{"kind":"reconcile","graph":"quick_Pointwise_nchw_fp32_Small.Small","entries":12,"passed":10,
 "failed":2,"filtered":0,"fail_numeric":1,"fail_build":0,"fail_execute":0,"no_catalog":1,
 "reconciled":true,"graph_outcome":"SWEPT","message":"","line":"reconcile graph=... ",
 "device_arch":"gfx1151","hipdnn_version":"1.2.3.4"}
```

`graph_outcome` is `SWEPT`, `SKIPPED` or `FAILED`. The same figures are printed per graph as

```
reconcile graph=quick_Pointwise_nchw_fp32_Small.Small entries=12 = passed=10 + failed=2 + filtered=0 [fail_numeric=1 fail_build=0 fail_execute=0 no_catalog=1]
```

**The identity `entries = passed + failed + filtered` is the proof that nothing was
dropped.** `entries` is counted when a catalog is read, the right-hand side as records are
written, so they are independent; `MISMATCH` is appended (and the graph fails) if they
differ. An engine with no catalog counts as one entry and as one `failed` (its
`NO_CATALOG`), so the identity covers it; so does a filtered engine with no catalog,
which is one `filtered` entry. The run ends with a summary of the same identity over all
graphs.

### Risks and limits

- **Time.** Engines that compile at plan-build time (the kernel ingestor with runtime
  compilation) compile every entry: hundreds of entries per graph is hours.
  Use `--catalog-shard`, `--catalog-filter` and `--gtest_filter`; `compile_time_ms` shows
  where the time goes. Every entry also refills the inputs, which for large graphs costs
  host time of its own.
- **Catalogs are not stable identifiers.** They depend on the graph, the device and the
  kernel packs shipped; compare reports by knob set, not by `entry_index`.
- **Reports are appended.** Re-running into an existing report adds lines; use a fresh
  path per run if you want one run per file.
- **Validated so far.** On gfx1151 the sweep has been run against `MIOPEN_ENGINE` and
  `HIP_MLOPS_ENGINE` (one-entry fallback catalogs over the quick tier) and against the
  in-tree test pointwise pack `hipkernel:Pointwise` (two-entry catalogs), plus a test-only
  engine with a kernel that writes no output (reported `FAIL_NUMERIC`). The rocKE engines
  have **not** been exercised: their shipped packs target gfx942/gfx950, so no rocKE
  catalog was fetched, no catalog sizes or collapsed entries were measured, and no rocKE
  graph was reconciled. Run a sweep on that hardware before relying on it for rocKE.
- **Not a CI test.** The sweep is slow, needs a fresh cache directory and a free device,
  and an engine that merely does not run a graph on this device is reported, not hidden.
- Support-claim sidecars are neither read nor enforced in sweep mode.

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
- `--test-config` points at a TOML file the provider owns (e.g.
  `config/MIOPEN_ENGINE.toml`) for per-test tolerance overrides and skips —
  see [Per-provider TOML config](#per-provider-toml-config-tolerance-overrides--skips).
- `TEST_CATEGORIES_YAML` generates tier-labelled ctest suites so
  `ctest -L quick|standard|...` selects tiers for the external run too.

Both the superbuild (target already present) and standalone provider builds
(`find_package`) are supported; if the package is not found the target is
skipped with a status message.

### Per-provider TOML config (tolerance overrides, validator overrides & skips)

Each provider owns one `--test-config` TOML file (e.g.
`miopen-provider/config/MIOPEN_ENGINE.toml`,
`hipblaslt-provider/config/HIPBLASLT_ENGINE.toml`,
`hip-kernel-provider/config/HIP_MLOPS_ENGINE.toml`) that can, without
recompiling or touching test source:

- **Override tolerances** for specific tests/groups, when that engine's
  numerics legitimately differ from the default atol/rtol (e.g. reduced
  precision from split-k accumulation).
- **Override the validator** for specific output tensors, when per-element
  allclose is the wrong *question* for that tensor rather than merely too
  tight. This is the only place a validator can be changed.
- **Skip tests** on specific architectures (or globally), when that engine
  has no applicable kernel/solution for a case.

```toml
[meta]
version = 1

[[tolerance_overrides]]
filters = ["Smoke/IntegrationGpuConvWrw3dBfp16.Correctness/14"]
atol = 1.19
rtol = 0.2

[[validator_overrides]]
filters       = ["*LayernormBackward*"]
tensors       = ["*::DSCALE", "*::DBIAS"]
validator     = "rms"
rms_threshold = 1e-4

[[test_skips]]
archs   = ["gfx90a", "gfx10", "gfx11", "gfx12"]   # optional; omit to skip everywhere
filters = ["*ConvFwdBiasActiv*"]
reason  = "ROCm/rocm-libraries#6979 — no engine has an applicable solution for ConvBiasActiv fusion"
```

- `filters` are GTest-style globs (`*` wildcard) matched against the full
  GTest name — same string a `--gtest_filter` would match.
- `tolerance_overrides`: later entries take precedence when multiple filters
  match. Both `atol` and `rtol` are required.
- `validator_overrides`: later entries take precedence. An entry applies only
  when a `filters` glob matches the test name **and** a `tensors` glob matches
  the output tensor's label — its name (e.g. `LayernormBackward_0::DSCALE`), or
  `uid=N` when the graph did not name it. Match on the tensor label rather than
  the uid: uids differ between a C++ graph test and the bundle captured from it,
  names do not. `validator` is `"allclose"`, `"allclose_matching_infinities"` or
  `"rms"`; `rms_threshold` is required and must be positive when the validator is
  `"rms"`, and must be absent for either of the other two — an entry that does not
  say exactly what it means is a load error, never a silent fall-back.
  `"allclose_matching_infinities"` grades exactly as `"allclose"` does, at the
  same resolved atol/rtol, except that an element that is infinite with the *same
  sign* in both the reference and the device output compares equal. NaN,
  opposite-signed infinities and finite-versus-infinite disagreements all still
  fail, and every finite element is still graded by atol/rtol. Use it for an
  output whose correct value is infinite on both sides — an SDPA forward
  log-sum-exp row that is fully masked is `-inf` in the reference and `-inf` on
  the device, and both are right, but `|ref - impl|` is NaN and plain allclose
  fails the tensor. It is **host-only**: there is no device implementation, so a
  tensor it selects fails with "Validator override NOT APPLICABLE ON DEVICE"
  whenever its comparison runs on the device — under `auto` that is every run
  where the GPU reference produced the expected values. Run such a config with
  `--validator cpu` (or `HIPDNN_TEST_VALIDATOR=cpu`), or narrow the `tensors`
  glob. Neither `"rms"` nor `"allclose_matching_infinities"` is
  defined for integer outputs (RMS has no integer formulation; an integer has no
  infinity to match): they are float, half, bfloat16 and double only, and a glob
  wide enough to catch an integer output fails that tensor with a message naming
  the glob to narrow.
  Absent any match the comparison is allclose — **allclose is the default
  everywhere, and this section is the only thing that changes it.** Use it when
  a per-element check is the wrong question, not to buy slack: an output that is
  a long reduction (layernorm/RMSNorm backward `dscale`/`dbias`) has elements
  that land arbitrarily near zero through cancellation, so per-element relative
  error is unbounded while the aggregate relative-RMS error is not. Prefer
  `tolerance_overrides` for everything else.
- `test_skips`: the first matching entry wins; `reason` is surfaced in the
  `GTEST_SKIP` message. `archs` (substring match against the raw
  `gcnArchName`) and `platforms` (`"windows"`/`"linux"`) are both optional —
  omit either to match any.
- Applies to **both** bundle/sweep tests and C++ graph tests; the lookup runs
  in the shared harness (`TestConfig`/`TestSettings`), not per test type.
  One caveat for `validator_overrides`: golden data validated against a *reference*
  executor (`BundleReferenceValidationHarness`) is always compared with allclose at
  the default tolerance. An engine's config describes how far that engine may
  drift; it never relaxes the gate on our own committed data.
- `[meta] version = 1` is required; the file is rejected on parse if missing
  or on an unsupported version.

Full schema and matching semantics: [`src/harness/TestSettings.hpp`](src/harness/TestSettings.hpp).

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
  rmsnorm, reduction, pointwise). Most of these are graph tests and are
  **not built by default** — see below.
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
reference," it belongs in a bundle instead. Convert it with
[`migration-scripts/`](migration-scripts/README.md): `--capture-bundles`
dumps the test's graph(s) as JSON, then `import_graph.py` merges each one
into the bundle tree — see ["Quick path: convert one existing C++
test"](migration-scripts/README.md#quick-path-convert-one-existing-c-test-no-full-pipeline-needed)
for the exact two-step commands. Don't add another parameterized
instantiation to the C++ test instead.

### Adding a C++ test: the CMake rule

The rule above is enforced by
[`src/integration-tests/CMakeLists.txt`](src/integration-tests/CMakeLists.txt).
Every `.cpp` under `src/integration-tests/` must be registered through exactly
one of two functions; a file registered through neither fails the configure
step as an *orphan*, so a new C++ test cannot slip into CI unnoticed and a
`target_sources()` call cannot quietly bypass the gate.

| Function | Builds when | Use for |
|---|---|---|
| `add_cpp_graph_test_sources(...)` | only `-DBUILD_CPP_GRAPH_TESTS=ON` | C++ graph-verification tests (the legacy style) |
| `add_always_built_test_sources(...)` | always | tests that have no bundle equivalent by construction |

`add_always_built_test_sources()` additionally requires the file to be listed
in `HIPDNN_IT_ALWAYS_BUILT_SOURCES` at the top of that same CMakeLists — a
central, reviewable edit — so exempting a test from the bundle-first rule
cannot happen inside a leaf directory. Adding a file there without listing it
is a `FATAL_ERROR` naming the file and pointing at bundles.

So, in practice:

- **New graph coverage** → add a bundle under `integration-test-bundles/`. Do
  not add a C++ file at all.
- **New non-graph test** (error path, API contract, round-trip, determinism) →
  `add_always_built_test_sources()` plus an entry in
  `HIPDNN_IT_ALWAYS_BUILT_SOURCES` with a comment saying why it can't be a
  bundle.
- **Debugging / bisecting against the bundle harness** → build your branch with
  `-DBUILD_CPP_GRAPH_TESTS=ON` to get the existing C++ graph tests back. They
  register under the `Smoke/*` GTest prefix that the provider
  `test_categories_integration.yaml` files still match, so they pick up CTest
  labels normally. Keep that pattern in the YAMLs even on a default `OFF`
  build: the always-built tests (`Smoke/IntegrationConvFwdSerializeRoundTripFp32`,
  `Smoke/IntegrationGpuResampleForward*`) register under it too.

One op is still exempt as migration debt: `resample/` has no bundles yet, so
`IntegrationGpuResampleForward.cpp` stays always-built rather than dropping the
op's only coverage. Remove it from the allow-list once its bundles land.

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
add_integration_test_target(hipdnn_my_new_op_tests ${CMAKE_CURRENT_BINARY_DIR})
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

## Bundle Tests

Bundle tests are data-driven: each test is a graph JSON + sweep of
shapes/dtypes/layouts — no C++ needed. The bundle runner discovers them
automatically from `integration-test-bundles/`.

### Searching cases

```bash
# Find all batchnorm bundle cases
python3 migration_scripts/find_case.py --op Batchnorm

# Find cases that have an epsilon input
python3 migration_scripts/find_case.py --input epsilon

# Find cases where epsilon is in [-1,1]
python3 migration_scripts/find_case.py --input epsilon:-1,1

# Full detail for a hashed case id
python3 migration_scripts/find_case.py --id f446b9 --detail
```

### Adding a bundle test

```bash
python3 migration_scripts/import_graph.py \
    --graph new_conv.json \
    --bundle-dir integration-test-bundles/
```

The case id is auto-generated and printed to stderr. No manual naming
needed. See [`migration_scripts/README.md`](migration_scripts/README.md)
for the full workflow and tooling reference.
## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Engine 'X' is not loaded` | Pass `--test-article /path/to/plugin.so`, or run from a superbuild |
| Bundle tests don't run | Check for `--no-bundles` / `HIPDNN_TEST_ALLOW_BUNDLES=0`; otherwise the data dir is missing (see the `Bundle tests are enabled but …` warning in the log) |
| A C++ graph test doesn't run | Expected — they build only with `-DBUILD_CPP_GRAPH_TESTS=ON`. See [Adding a C++ test](#adding-a-c-test-the-cmake-rule) |
| Tests can't find bundle data | `dvc pull` the op, or run with `--verification-mode gpu` to skip golden comparison |
| Smoke tier timing out | A shape is missing its tier prefix — check `INSTANTIATE_TEST_SUITE_P` prefixes |
| `No tests matched the filter` | Use a single `-` for negative filters: `-Standard*:Comprehensive*:Full*` |

## See Also

- [`integration-test-bundles/README.md`](integration-test-bundles/README.md) —
  on-disk bundle layout, DVC remotes, and add/update/remove/pull/push workflow.
- [`migration-scripts/README.md`](migration-scripts/README.md) — capture →
  place → verify pipeline, `import_graph.py`, `find_case.py`, and the C++ → bundle
  field mapping.
- [RFC 0011 — Golden Reference Validation](../../projects/hipdnn/docs/rfcs/0011_GoldenReferenceValidation.md)
  — the bundle/sweep naming spec (§4.1) and design rationale.
- [`docs/support-claim-enforcement.md`](docs/support-claim-enforcement.md) —
  `.support.json` sidecars, the verdict set comparison, the `TestBody()`
  enforcement lifecycle, and how to read the claim summary.

## Project policies

This suite is part of the hipDNN project. Shared project documentation and
policies are maintained in hipDNN:

- [hipDNN Overview](../../projects/hipdnn/README.md)
- [Contributing Guidelines](../../projects/hipdnn/CONTRIBUTING.md)
- [Security Policy](../../projects/hipdnn/SECURITY.md)
