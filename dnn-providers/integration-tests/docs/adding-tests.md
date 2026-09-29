# Adding Tests and Updating Claims

How to add coverage to the cross-provider suite and keep everything that
describes it — metadata, support claims, golden data, engine config — in step.
Terms such as *cell*, *lane* and *oracle* are defined in the
[index](README.md#terms).

- The files touched here are described in [File Formats](file-formats.md).
- Running what you added, and reading the result, is in
  [Running the Tests](running-tests.md).

Commands run from the repository root; the superbuild is in `build/`.

## Pick the mechanism first

| | **Bundle** (default) | **C++ integration test** (special cases) |
|---|---|---|
| What it is | Graph JSON (single graph or sweep), metadata, a support sidecar, optional golden tensors | `buildGraph()` plus `INSTANTIATE_TEST_SUITE_P` |
| Add a case | Run a tool; no compile | Write C++, recompile |
| Built and run by default | yes | only listed files — see [the CMake rule](#c-tests-the-cmake-rule) |
| Use for | "does this graph run on this engine and match a reference" | anything else: error paths, API contracts, serialization round-trips, determinism, benchmarking knobs |

**New graph-verification coverage must be a bundle.** If a proposed C++ test is
really "build graph X, run it, compare", make it a bundle instead.

Then pick the bundle kind:

- **Sweep** — the same topology across several shapes, dtypes or layouts. The
  common case; a new shape is one more case, not a new directory.
- **Single graph** — exactly one concrete graph with nothing to vary, a graph
  you want pinned byte-for-byte, or cases whose *structure* differs (a sweep
  can vary values, not node count or wiring).

When in doubt, `import_graph.py` decides: it appends to a sweep whose skeleton
matches and creates a new one otherwise.

Whatever the kind, a bundle is not complete until it has all three required
parts ([formats](file-formats.md#map)):

1. **The graph** — `{Name}.json`, or `graph.template.json` plus a `sweep.json` case.
2. **Metadata** — `{Name}.meta.json`, or `cases[].metadata` on every sweep case,
   with `format_version`, `generator` and `reference_source`.
3. **A support-claim sidecar** — `{Name}.support.json`, or the sweep's
   `support.json` — even if it claims nothing yet.

Golden data is the only optional part.

## End to end: a bundle for a new op

The whole path, in order; each step links to its details.

1. **Get graph JSON.** From an existing C++ graph test, capture it. For an op
   with no test at all, write a parameterized C++ graph test first, then capture
   it — see [A brand-new sweep](#a-brand-new-sweep).
2. **Import it** with `import_graph.py`
   ([details](#from-a-graph-you-already-have)). This writes the graph, the case
   metadata and an empty `support.json`, and prints the GTest name.
3. **Check the oracle.** If no reference executor supports the op, decide
   between golden data and a lower `enforcement_level`
   ([details](#when-no-reference-executor-supports-the-op)).
4. **Re-run CMake**, so the build's copy of the bundle tree picks up the new
   files ([why](running-tests.md#the-build-reads-a-copy-of-the-bundles)).
5. **Run the lane** for each engine you can, with the output visible
   (`ctest ... -V`, or the check target), and read both summaries.
6. **Record claims** for the engines that accept the graph
   ([details](#record-claims-with---write-support-claims)), then re-run CMake and
   the lane to confirm them.
7. **Verify and commit** — run the checklist in
   [Before you open the PR](#before-you-open-the-pr).

## Add a bundle

### From a graph you already have

```bash
python3 dnn-providers/integration-tests/migration-scripts/import_graph.py \
    --graph new_conv.json \
    --bundle-dir dnn-providers/integration-tests/integration-test-bundles \
    --tier quick \
    --meta reference_source="where this graph came from"
```

It hashes the graph's skeleton and then:

1. **Exact duplicate** (same graph, seed and inputs) → prints `DUPLICATE`, writes
   nothing. `--strict` makes that exit non-zero; `--force` appends anyway.
2. **New case for an existing topology** → appends one case to that sweep's
   `sweep.json`, leaving every existing case untouched. If no matching sweep is
   in `--tier`, it appends to a match in another tier — the printed path and
   GTest name say where it went.
3. **New topology** → creates a new sweep directory under `{tier}/{Op}/`. It
   never writes into a directory that already holds a sweep: if the preferred
   name is taken it uses the next free `Variant2`, `Variant3`, ….

Either way it prints the case id and the GTest name the harness will register
(`gtest name: quick_Relu_Nchw.2_3_4_5_fp32_nchw`), sets `generator` to
`import_graph.py` unless you pass `--meta generator=…`, warns if there is no
`reference_source`, and creates an empty `support.json` in the sweep directory
if it has none (an existing one is never touched). Other flags: `--seed`,
`--meta key=value` (repeatable; `inputs` takes JSON), `--dry-run`.

Review `git status` and `git diff` afterwards: expect one new case, or one new
directory with three files.

### From an existing C++ graph test

Capture its graphs, then import each one. Capture needs a build with
`-DBUILD_CPP_GRAPH_TESTS=ON`, since the C++ graph tests are not compiled in
otherwise.

```bash
build/bin/hipdnn_integration_tests --capture-bundles /tmp/captured \
    --gtest_filter='Full/IntegrationGpuConvFwdBiasActiv2dFp16.Correctness/*'

find /tmp/captured -name '*.json' ! -name '*.meta.json' | while read -r graph; do
    python3 dnn-providers/integration-tests/migration-scripts/import_graph.py \
        --graph "$graph" \
        --bundle-dir dnn-providers/integration-tests/integration-test-bundles \
        --meta reference_source="c++ integration suite: $(basename "$(dirname "$graph")")"
done
```

Capture writes `{suite}/{case}/{case}.json` plus a `.meta.json` beside each
graph; the suite name keeps its `/` (`Full/IntegrationGpu…`), so files sit a
directory deeper than it looks — hence `find`. `import_graph.py` does not read
the capture's `.meta.json`; pass `--seed` and `--meta inputs=<json>` yourself if
the imported case must reproduce the C++ test's inputs exactly. Once each
printed test runs and passes as a bundle, delete the C++ registration. The bulk
pipeline and its byte-level verification are in
[`migration-scripts/README.md`](../migration-scripts/README.md).

### A brand-new sweep

There is no tool that generates a sweep's case matrix from a list of shapes,
and cases are not authored by hand. The route is a round trip through C++:

1. Write the matrix as a parameterized C++ graph test in
   `src/integration-tests/<op>/`, following an existing one (for example
   `src/integration-tests/rmsnorm/IntegrationGpuRMSNorm.cpp`), and register it
   with `add_cpp_graph_test_sources()` in that directory's `CMakeLists.txt`.
2. Build with `-DBUILD_CPP_GRAPH_TESTS=ON` and run it until it passes.
3. Capture and import it as above.
4. Delete the C++ test once the bundle cases run and pass.

### When no reference executor supports the op

An engine's output is only checked if an oracle exists
([Verification modes](running-tests.md#verification-modes)). Which ops each
reference executor must handle is listed in
`src/harness/bundle/ReferenceOpCoverage.hpp`. If yours is not there, the bundle
lands in `UNVERIFIABLE BUNDLES` and nothing checks it. Either:

- **add golden data** produced by an independent implementation (see
  [Golden data with DVC](#golden-data-with-dvc)), or
- **set `enforcement_level`** in the metadata to `applicability` (the engine
  accepts the graph) or `buildable` (its plans compile), so the bundle claims
  only what can be checked today.

### Finding existing cases

```bash
python3 dnn-providers/integration-tests/migration-scripts/find_case.py --op Batchnorm
python3 dnn-providers/integration-tests/migration-scripts/find_case.py --id f446b9 --detail   # includes the --gtest_filter
```

More queries (`--dtype`, `--layout`, `--input`, `--shape`) are in
[`migration-scripts/README.md`](../migration-scripts/README.md).

### Choosing a tier

Put the bundle under the cheapest tier that still catches what it is for.
`quick` runs in every provider lane's PR tier, so large shapes belong in
`standard` or higher. The directory *is* the tier; what each tier runs is in
[Test tiers](running-tests.md#test-tiers).

## Golden data with DVC

### Golden data is optional

A bundle with no golden data is still a full test: its output is compared
against the GPU or CPU reference executor. Add golden data when you want the
stricter golden comparison, or when no reference executor can run the op.

Run DVC commands from the repository root. Reads are anonymous; `dvc push`
needs AWS credentials. After any pull, re-run CMake so the build's copy has the
data.

```bash
dvc pull                                                                           # everything
dvc pull -R dnn-providers/integration-tests/integration-test-bundles/quick/SdpaFwd # one op
```

### Add golden data to a single-graph bundle

```bash
BUNDLE=dnn-providers/integration-tests/integration-test-bundles/quick/ConvFwd/nhwc/fp16/resnet50_layer3

# 1. Put resnet50_layer3.json, resnet50_layer3.meta.json, resnet50_layer3.support.json
#    (empty if nothing is claimed yet) and resnet50_layer3.tensor<uid>.bin in $BUNDLE.
# 2. Write one pointer listing every .bin. New hipDNN ops route to the golden-data remote.
{ echo "outs:"; for f in "$BUNDLE"/*.tensor*.bin; do
    echo "- path: $(basename "$f")"; echo "  remote: golden-data"; done; } \
    > "$BUNDLE/resnet50_layer3.tensors.dvc"
# 3. Let DVC fill in hashes and cache the data.
dvc commit -f "$BUNDLE/resnet50_layer3.tensors.dvc"
# 4. Commit everything except the .bin files, then upload the data.
git add "$BUNDLE/resnet50_layer3.json" "$BUNDLE/resnet50_layer3.meta.json" \
        "$BUNDLE/resnet50_layer3.support.json" "$BUNDLE/resnet50_layer3.tensors.dvc"
git commit -m "Add ConvFwd resnet50_layer3 bundle"
dvc push -r golden-data
```

`dvc push` uploads from the local **cache**, not the working tree — always
`dvc commit` first, or the push silently skips the new files. Legacy ops whose
pointers carry no `remote:` key use the default `storage` remote (`dvc push`
with no `-r`).

### Add golden data to a sweep case

Put the files in `golden/{CaseId}/` (`tensor<uid>.bin` plus a `tensors.dvc`
pointer written as above), then add `"golden": {"path":
"golden/{CaseId}/tensors.dvc"}` to that case in `sweep.json` — editing an
existing case's fields by hand is fine.

### Update or remove golden data

- **Update:** overwrite the `.bin` files, `dvc commit -f` the pointer (re-author
  its `outs:` list first if the set of files changed), commit the pointer, `dvc push`.
- **Remove a bundle:** `dvc remove` its pointer, delete the directory, commit.
- **Roll back:** revert the pointer in git; `dvc pull` then fetches the old data,
  which stays in S3 by content hash.

### Generated bundles

Some ops are produced by a generator rather than captured, for example
`integration-test-bundles/quick/SdpaFwd/generate_golden_data.sh` and the
generators in `reference-data-scripts/`. Regenerate, then:

```bash
dvc commit -f -R <op dirs>
dvc push -r golden-data -R <op dirs>
git add <op dirs>          # .json, .meta.json, .support.json, .tensors.dvc — .bin is ignored
```

`dvc commit` keeps each pointer's existing `remote:` key. A generator that does
not write a support sidecar leaves that step to you.

### Check the data itself

```bash
python dnn-providers/integration-tests/reference-data-scripts/verify_golden_bundles.py \
    --require-data dnn-providers/integration-tests/integration-test-bundles
build/bin/hipdnn_golden_data_tests --reference cpu
```

The first checks graph JSON, metadata (including the required `generator` and
`reference_source`), tensor sizes, and NaN/Inf in outputs; `--require-data`
makes a pointer whose `.bin` was not pulled an error instead of a warning. The
second recomputes each golden bundle with a reference executor and compares.

### DVC troubleshooting

| Symptom | Fix |
|---|---|
| `dvc push` auth error | Check `aws sts get-caller-identity`; writes need AWS credentials, reads do not. |
| A `.dvc` pointer exists but no `.bin` on disk | `dvc pull path/to/Name.tensors.dvc` |
| A `.bin` was committed to git by accident | `git rm --cached` it, then `dvc commit -f` its pointer. |
| Tensor files were added or removed | Re-author the pointer's `outs:` list, then `dvc commit -f` it. |
| Tests do not see new data | Re-run CMake; then `dvc status` to check for drift between pointers and the cache. |

## Updating support claims

Support-claim sidecars ([format](file-formats.md#support-claim-sidecars--namesupportjson-and-supportjson))
record which engines accept which graphs on which arch and platform. They are
**written by the harness, reviewed by people**.

### When to update them

| Trigger | Action |
|---|---|
| A run's `SUPPORT CLAIM SUMMARY` lists entries under **`unclaimed_support`** | The engine accepts graphs nobody has claimed. Record them. This is the main trigger; it never fails a run, so look for it. |
| You added bundles | Record claims for the engines and archs you can run; if none accepts the graph, keep the [empty sidecar](#when-no-engine-accepts-the-graph-yet). |
| `verdicts` all zero while `unenforced.no_applicable_claim` > 0 | This arch/platform has no claims yet (typical on a bring-up ASIC); record them. |
| `claim_failures` → `CLAIM_BROKEN`, and dropping the support is **intended** | [Retract the claim](#retract-a-claim). Otherwise fix the engine. |
| `failed_in_use` | A claimed cell the engine gets wrong. Fix the engine, or retract the claim until it is fixed. |

Only bundles that already have a sidecar can report `unclaimed_support`; a
bundle with none is silent.

An `unclaimed_support` entry names the bundle and the sweep cases; the
summary's `run` block names the engine, arch and platform it was seen on:

```json
"run": { "arch": "gfx942", "engine": "MIOPEN_ENGINE", "platform": "linux" },
"unclaimed_support": [
  { "bundle": "integration-test-bundles/quick/ConvFwd/sweep.json",
    "cases": ["case_a", "case_b"], "reached": "verified", "required": "verified" }
]
```

Look at `reached` before claiming. `reached` equal to `required` means the run
verified the result; `reached` below `required` means the engine accepts the
graph but nothing checked its output — the writer below will claim it anyway,
so decide first whether you want it claimed.

### Record claims with `--write-support-claims`

On a machine with the arch and platform you want to claim:

```bash
build/bin/hipdnn_integration_tests \
    --test-article <path to the provider plugin> \
    --test-engine  MIOPEN_ENGINE \
    --test-config  dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml \
    --golden-data-dir dnn-providers/integration-tests/integration-test-bundles \
    --gtest_filter='quick_*:standard_*:comprehensive_*:full_*' \
    --write-support-claims
```

- **Point `--golden-data-dir` at the source tree.** The writer edits sidecars in
  place; the build's copy is overwritten on the next configure.
- **Pass `--test-engine`.** Without it the writer records every engine in the
  plugin (hip-kernel-provider has two).
- **Pass the engine's `--test-config`**, as the lane does. Cases the TOML skips
  are skipped before observation, so you do not claim what the lane never runs.
- **Filter to bundles** (as above, or narrower). The writer observes bundles
  only; any C++ tests in the binary still run normally and can fail the run.
  Graphs a filter removes keep their claims as they were.
- Run one writer at a time per bundle tree; concurrent writers race.

What it does: for each bundle it asks for the ranked list, records the engine
**as supporting the graph if it is ranked**, and skips the test (`support-claim
authoring run (--write-support-claims)`). It does not execute or compare the
bundle. It only ever **adds** claims for the running machine's arch token and
platform, merges them into existing sidecars, writes them in the writer's
normal form ([details](file-formats.md#support-claim-sidecars--namesupportjson-and-supportjson)),
and leaves no git diff when nothing changed. It does not create a sidecar for a
graph no engine accepted.

It prints a summary instead of the claim summary:

```text
==== SUPPORT CLAIM WRITE SUMMARY ====
  graphs registered: N  observed: N  not observed: N  skipped in SetUp: N  unaccounted for: N
  observations: N  written: N  unchanged: N  skipped: N  errors: N
```

It exits 1 when any test in the run failed, when zero tests were selected, when
nothing was observed at all, when any sidecar could not be written, when a
graph reached the engine but could not be observed (it failed to open, or the
query did not resolve), or when registered graphs are unaccounted for without a
`--gtest_filter` or shard split to explain them. Graphs removed by a filter, or
skipped in `SetUp()`, are reported and their claims left as they were.

### Confirm, review, commit

1. **Re-run CMake**, so the lane reads the sidecars you just wrote (or run the
   binary with `--golden-data-dir` pointing at the source tree).
2. **Run the lane enforcing** (no `--write-support-claims`) and read the claim
   summary. The new cells should no longer appear under `unclaimed_support`,
   none should be under `failed_in_use`, and `verdicts.confirmed` should have
   grown. `confirmed` and `accepted` are counts only: a claimed cell that
   stayed `accepted` was never verified — find it under `UNVERIFIABLE BUNDLES`.
3. **Review the diff.** Expect additions only; the writer never removes a claim.
4. **Verify and commit:**

   ```bash
   python dnn-providers/integration-tests/scripts/verify_support_claims.py
   ```

   Run it by hand: the pre-commit configuration currently hides
   `integration-test-bundles/` from every hook, including this one.

Each machine can only claim its own cell. Claims for other archs or platforms
come from runs on those machines.

### When no engine accepts the graph yet

A bundle still needs a sidecar when no engine you can run accepts it — a new op
whose engine support has not landed, or a graph for hardware you do not have.
`import_graph.py` creates this empty sidecar for you; otherwise commit it by
hand, exactly as shown (note the trailing newline):

```json
{
  "claims": {},
  "version": 1
}
```

It records that the bundle was reviewed and nothing is claimed yet. Once an
engine accepts the graph, a run reports it under `unclaimed_support` and
`--write-support-claims` fills the claims in.

### Retract a claim

Retraction is always a deliberate, reviewed edit — no tool removes a claim,
because a regression and an intended withdrawal look identical to the harness.

1. Delete the engine's arch/platform entry (or move the sweep case out of its
   group) by hand, keeping the file in the writer's normal form.
2. If the engine still *accepts* the graph (it is broken in use, not declined),
   also add a `[[test_skips]]` entry for that case and arch to the engine's TOML.
   Otherwise the next `--write-support-claims` run observes it as supported and
   adds the claim straight back.
3. Say in the PR why the engine no longer supports it, and link the tracking
   issue.

## Record an engine's known limitation

Edit the engine's `config/<ENGINE_NAME>.toml`
([format](file-formats.md#per-engine-test-config--configengine_nametoml)):

- **The engine has no kernel for a case on some arch** → `[[test_skips]]` with
  `archs`/`platforms` as narrow as the evidence, and a `reason` that links the
  tracking issue. The skip is visible in every run as `[arch …] <reason>`.
- **The engine's numerics legitimately differ** → `[[tolerance_overrides]]`
  for the affected tests only.
- **Per-element comparison is the wrong question for an output** (long
  reductions whose elements cancel toward zero) → `[[validator_overrides]]`
  with `validator = "rms"` for those tensors only.
- **An output is correctly infinite on both sides** (a fully masked SDPA row's
  log-sum-exp) → `[[validator_overrides]]` with
  `validator = "allclose_matching_infinities"` for those tensors only. It is
  host-only: the comparison must run on the host (`--validator cpu` or
  `HIPDNN_TEST_VALIDATOR=cpu`), or the tensor fails.

Do not use a skip to hide a regression on a claimed cell unless you are also
retracting the claim; the claim is there to make that regression visible.

For arch-wide exclusions of whole suites, the tier YAML's `exclude_gpu` block is
the coarser tool; a `"*"` pattern disables the suite on that arch entirely and
should carry a comment and a tracking issue.

## Add an engine lane

A new engine joins the suite by registering a lane in its provider's
`CMakeLists.txt`. The suite installs a CMake package, `hipdnn_integration_tests`,
exporting the binary and `cmake/HipdnnIntegrationTestHelpers.cmake`:

```cmake
if(NOT TARGET hipdnn_integration_tests)
    find_package(hipdnn_integration_tests CONFIG QUIET)   # standalone provider build
endif()

if(TARGET hipdnn_integration_tests)
    add_external_integration_test_target(
        TARGET_NAME          ${PROJECT_NAME}-external-integration-check
        PLUGIN_TARGET        miopen_plugin                          # --test-article
        ENGINE_NAME          MIOPEN_ENGINE                          # --test-engine
        INSTALL_SUBDIR       miopen_plugin
        TEST_CONFIG          ${CMAKE_CURRENT_SOURCE_DIR}/config/MIOPEN_ENGINE.toml
        TEST_CATEGORIES_YAML ${CMAKE_CURRENT_SOURCE_DIR}/test_categories_integration.yaml
        TEST_NAME_PREFIX     ${PROJECT_NAME}-external-integration  # CTest suite prefix
    )
endif()
```

That creates the check target and, from the YAML, one CTest suite per tier
running
`hipdnn_integration_tests --test-article <plugin> --test-engine <ENGINE> --test-config <toml> --gtest_filter=<tier patterns>`.
Every lane names `--test-engine`, so every lane enforces support claims. The
function also accepts `REFERENCE_EXECUTOR`, `GTEST_FILTER` (applied to the
check target and to a lane without a tier YAML, not to the tier suites),
`ENVIRONMENT`, `ENVIRONMENT_MODIFICATION`,
`FIXTURES_REQUIRED`, `INSTALL_TEST_FILE` and `INSTALL_ENVIRONMENT`. Without
`TEST_CATEGORIES_YAML` the lane is one CTest test with no tier labels. Without
the package the lane is skipped at configure time.

A new lane also needs its engine TOML and tier YAML
([formats](file-formats.md#per-engine-test-config--configengine_nametoml)); copy
an existing provider's and keep the same tier shape.

## C++ tests: the CMake rule

Every `.cpp` under `src/integration-tests/` must be registered through exactly
one of two functions, or configure fails listing it as an orphan:

| Function | Builds | Use for |
|---|---|---|
| `add_cpp_graph_test_sources(...)` | only with `-DBUILD_CPP_GRAPH_TESTS=ON` | C++ graph tests |
| `add_always_built_test_sources(...)` | always | tests that cannot be a bundle |

`add_always_built_test_sources()` also requires the file to be listed in
`HIPDNN_IT_ALWAYS_BUILT_SOURCES` at the top of
`src/integration-tests/CMakeLists.txt`, with a comment saying why it cannot be a
bundle. That list is the one central, reviewable place a new C++ test can enter
CI.

To run in the provider lanes' PR tiers, a C++ test must be instantiated under
a `Smoke`, `Quick` or `Standard` prefix; a test with no tier prefix runs only
from `comprehensive` up ([Test tiers](running-tests.md#test-tiers)).

Tests of behavior specific to one plugin belong in that provider's own
`integration_tests/` directory (for example
`dnn-providers/miopen-provider/integration_tests/`), not in the shared suite.

C++ test names are checked by the `*_test_name_validation` CTest entries: a test
**case** name must not contain the suite keywords `Test`, `Integration`, `Gpu`,
or a datatype token such as `Fp16`.

## Reference-executor tests

`tests/gpu-ref/` holds the tests for the GPU/CPU reference executors themselves
(`hipdnn_gpu_ref_tests`), not for any engine. To add one:

1. Add the `.cpp` to the `add_executable(hipdnn_gpu_ref_tests …)` list in
   `tests/CMakeLists.txt`; a file not listed there is never compiled.
2. Follow the shape-catalog pattern in `tests/gpu-ref/ConvShapeCatalog.hpp`.
3. Define all four tiers and name cases by shape tag (the case-list functions
   below are placeholders for your catalog's):

   ```cpp
   INSTANTIATE_TEST_SUITE_P(Smoke,         MyNewOp2dTestFp32, ::testing::ValuesIn(getSmallCases()),      byTag());
   INSTANTIATE_TEST_SUITE_P(Standard,      MyNewOp2dTestFp32, ::testing::ValuesIn(getMediumCases()),     byTag());
   INSTANTIATE_TEST_SUITE_P(Comprehensive, MyNewOp2dTestFp32, ::testing::ValuesIn(getLargeEdgeCases()),  byTag());
   INSTANTIATE_TEST_SUITE_P(Full,          MyNewOp2dTestFp32, ::testing::ValuesIn(getLargeStressCases()), byTag());
   ```

New shapes added to an existing catalog are picked up automatically. Some
existing files use `Quick` rather than `Smoke` for the first tier — match the
file you extend. The other files directly under `tests/` are the harness's own
unit tests (`hipdnn_integration_tests_unit_tests`).

## Before you open the PR

- [ ] New graph coverage is a bundle, not a C++ test.
- [ ] Every new bundle has its graph, its metadata (with `generator` and
      `reference_source`) and a support sidecar, even an empty one.
- [ ] You re-ran CMake after changing bundles, sidecars or pulling data, then ran
      the lane for each engine you touched with its output visible, and read its
      `TEST COVERAGE SUMMARY` and `SUPPORT CLAIM SUMMARY` — not just the exit code.
- [ ] `unclaimed_support` for your new bundles is recorded (or deliberately left
      unclaimed), and no new claim is in `failed_in_use`.
- [ ] Golden data, if any, is `dvc commit`ted and `dvc push`ed; only `.json` and
      `.dvc` files are in git.
- [ ] `python dnn-providers/integration-tests/scripts/verify_support_claims.py`
      passes, run by hand.
- [ ] If you regenerated or renamed sweep cases, no sidecar names a case that no
      longer exists, and miopen-provider's `ffm-quick` ids still exist
      (`ctest --test-dir build/dnn-providers/miopen-provider -L ffm-quick -N`).
