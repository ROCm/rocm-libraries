# Adding Tests and Updating Claims

How to add coverage to the cross-provider suite and keep everything that
describes it — golden data, support claims, engine config — in step.

- The files touched here are described in [File Formats](file-formats.md).
- Running what you added, and reading the result, is in
  [Running the Tests](running-tests.md).

## Pick the mechanism first

| | **Bundle** (default) | **C++ integration test** (special cases) |
|---|---|---|
| What it is | Graph JSON, optionally a sweep of cases and golden tensors | `buildGraph()` plus `INSTANTIATE_TEST_SUITE_P` |
| Add a case | Run a tool; no compile | Write C++, recompile |
| Built and run by default | yes | only listed files — see [the CMake rule](#c-tests-the-cmake-rule) |
| Use for | "does this graph run on this engine and match a reference" | anything else: error paths, API contracts, serialization round-trips, determinism, benchmarking knobs |

**New graph-verification coverage must be a bundle.** If a proposed C++ test is
really "build graph X, run it, compare", convert it to a bundle instead of
adding another instantiation.

Then pick the bundle kind:

- **Template sweep** — the same topology across several shapes, dtypes or
  layouts. The common case; a new shape is one more case, not a new directory.
- **Single graph** — exactly one concrete graph with nothing to vary, a graph
  you want pinned byte-for-byte, or cases whose *structure* differs (a sweep
  can vary values, not node count or wiring).

When in doubt, `import_graph.py` decides: it appends to a sweep whose skeleton
matches and falls back to a new bundle otherwise.

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
2. **New case for an existing topology** → appends to that `sweep.json`.
3. **New topology** → creates a new `graph.template.json` + `sweep.json`.

The generated case id is printed to stderr; the full test name is
`{tier}_{Op}_{Topology}.{id}`. Other flags: `--seed`, `--meta key=value`
(repeatable), `--dry-run`.

### From an existing C++ graph test

Capture its graphs, then import each one:

```bash
./build/bin/hipdnn_integration_tests --capture-bundles /tmp/captured \
    --gtest_filter='Full/IntegrationGpuConvFwdBiasActiv2dFp16.Correctness/*'

for graph in /tmp/captured/*/*/*.json; do
    [[ "$graph" == *.meta.json ]] && continue
    python3 dnn-providers/integration-tests/migration-scripts/import_graph.py \
        --graph "$graph" \
        --bundle-dir dnn-providers/integration-tests/integration-test-bundles \
        --meta reference_source="c++ integration suite: $(basename "$(dirname "$graph")")"
done
```

The capture needs a build with `-DBUILD_CPP_GRAPH_TESTS=ON` (otherwise the C++
graph tests are not compiled in). `import_graph.py` does not read the
`.meta.json` that capture writes next to each graph; pass `--seed` and
`--meta inputs=<json>` yourself if the imported case must reproduce the C++
test's inputs exactly. Once each printed test name runs and passes as a bundle,
delete the C++ registration. The bulk pipeline and its byte-level verification
are in [`migration-scripts/README.md`](../migration-scripts/README.md).

### A brand-new sweep

There is no tool that generates a new sweep's case matrix from a list of
shapes. **Do not hand-write `sweep.json`.** The current route is a round trip:
write the matrix as a parameterized C++ graph test (fast to iterate), capture
it, import it as above, then delete the C++ test.

### Finding existing cases

```bash
python3 dnn-providers/integration-tests/migration-scripts/find_case.py --op Batchnorm
python3 dnn-providers/integration-tests/migration-scripts/find_case.py --dtype fp16 --layout nhwc
python3 dnn-providers/integration-tests/migration-scripts/find_case.py --input epsilon:-1,1
python3 dnn-providers/integration-tests/migration-scripts/find_case.py --id f446b9 --detail   # includes the --gtest_filter
```

### Choosing a tier

Put the bundle under the cheapest tier that still catches what it is for.
`quick` runs on every PR in every provider lane, so large shapes belong in
`standard` or higher. The directory *is* the tier; see
[Test tiers](running-tests.md#test-tiers).

### A graph-only bundle is complete

A bundle with no golden data is still a full test: its output is compared
against the GPU or CPU reference executor. Add golden data only when you want
the stricter golden comparison, or when no reference executor can run the op.

## Golden data with DVC

Run DVC commands from the repository root. Reads are anonymous; `dvc push`
needs AWS credentials.

```bash
dvc pull                                                                        # everything
dvc pull dnn-providers/integration-tests/integration-test-bundles/quick/SdpaFwd # one op
```

### Add golden data to a single-graph bundle

```bash
BUNDLE=dnn-providers/integration-tests/integration-test-bundles/quick/ConvFwd/nhwc/fp16/resnet50_layer3

# 1. Put {Name}.json and {Name}.tensor<uid>.bin in $BUNDLE.
# 2. Write one pointer listing every .bin. New hipDNN ops route to the golden-data remote.
{ echo "outs:"; for f in "$BUNDLE"/*.tensor*.bin; do
    echo "- path: $(basename "$f")"; echo "  remote: golden-data"; done; } \
    > "$BUNDLE/resnet50_layer3.tensors.dvc"
# 3. Let DVC fill in hashes and cache the data.
dvc commit -f "$BUNDLE/resnet50_layer3.tensors.dvc"
# 4. Commit the JSON and the pointer (never the .bin), then upload the data.
git add "$BUNDLE/resnet50_layer3.json" "$BUNDLE/resnet50_layer3.tensors.dvc"
git commit -m "Add ConvFwd resnet50_layer3 bundle"
dvc push -r golden-data
```

`dvc push` uploads from the local **cache**, not the working tree — always
`dvc commit` first, or the push silently skips the new files. Legacy ops whose
pointers carry no `remote:` key use the default `storage` remote (`dvc push`
with no `-r`).

For a sweep case the same applies per case: `golden/{CaseId}/tensor<uid>.bin`
plus `golden/{CaseId}/tensors.dvc`, and the case's `"golden": {"path":
"golden/{CaseId}/tensors.dvc"}` in `sweep.json`.

### Update or remove golden data

- **Update:** overwrite the `.bin` files, `dvc commit -f` the pointer (re-author
  its `outs:` list first if the set of files changed), commit the pointer, `dvc push`.
- **Remove a bundle:** `dvc remove` its pointer, delete the directory, commit.
- **Roll back:** revert the pointer in git; `dvc pull` then fetches the old data,
  which stays in S3 by content hash.

### Generated bundles

Some ops are produced by a generator rather than captured, for example
`integration-test-bundles/quick/SdpaFwd/generate_golden_data.sh` and the
generators in `reference-data-scripts/`. Regenerate, then
`dvc commit -f --recursive <op dirs>` and `dvc push -r golden-data --recursive
<op dirs>`; `dvc commit` keeps each pointer's existing `remote:` key.

### Check the data itself

```bash
python dnn-providers/integration-tests/reference-data-scripts/verify_golden_bundles.py \
    dnn-providers/integration-tests/integration-test-bundles
./build/bin/hipdnn_golden_data_tests --reference cpu
```

The first checks JSON, tensor sizes, metadata, and NaN/Inf in outputs. The
second recomputes each golden bundle with a reference executor and compares.

### DVC troubleshooting

| Symptom | Fix |
|---|---|
| `dvc push` auth error | Check `aws sts get-caller-identity`; writes need AWS credentials, reads do not. |
| A `.dvc` pointer exists but no `.bin` on disk | `dvc pull path/to/Name.tensors.dvc` |
| A `.bin` was committed to git by accident | `git rm --cached` it, then `dvc commit -f` its pointer. |
| Tensor files were added or removed | Re-author the pointer's `outs:` list, then `dvc commit -f` it. |
| Tests do not see new data | `dvc status` to check for drift between pointers and the cache. |

## Updating support claims

Support-claim sidecars ([format](file-formats.md#support-claim-sidecars--namesupportjson-and-supportjson))
record which engines accept which graphs on which arch and platform. They are
**written by the harness, reviewed by people**.

### When to update them

| Trigger | Action |
|---|---|
| A run's `SUPPORT CLAIM SUMMARY` lists entries under **`unclaimed_support`** | The engine accepts graphs nobody has claimed. Record them — this is the main trigger. |
| You added bundles | New bundles start with no claims; add them for the engines and archs you can run. |
| `unenforced.no_applicable_claim` covers every claim-bearing graph | This arch/platform has no claims yet (typical on a bring-up ASIC); record them. |
| An engine gained support for an op | The enforcing lane will show it as `unclaimed_support`; record it. |
| `CLAIM_BROKEN` and dropping the support is **intended** | Retract the claim by hand — see below. Otherwise fix the engine. |
| `failed_in_use` | Do **not** claim that cell. The engine accepts the graph but gets it wrong. |

An `unclaimed_support` entry names the bundle, the sweep cases, and — in the
summary's `run` block — the engine, arch and platform it was seen on:

```json
"run": { "arch": "gfx942", "engine": "MIOPEN_ENGINE", "platform": "linux" },
"unclaimed_support": [
  { "bundle": "integration-test-bundles/quick/ConvFwd/sweep.json",
    "cases": ["case_a", "case_b"], "reached": "verified", "required": "verified" }
]
```

`reached` says how far the run got. An entry that reached its `required` depth
is a stronger case for claiming than one the engine merely ranked.

### Record claims with `--write-support-claims`

On a machine with the arch and platform you want to claim, from the repository
root:

```bash
./build/bin/hipdnn_integration_tests \
    --test-article /path/to/libmiopen_plugin.so \
    --test-engine  MIOPEN_ENGINE \
    --test-config  dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml \
    --golden-data-dir dnn-providers/integration-tests/integration-test-bundles \
    --write-support-claims
```

- **Point `--golden-data-dir` at the source tree.** The writer edits sidecars in
  place; a build-tree copy is lost on the next clean build.
- **Pass `--test-engine`.** Without it the writer records every engine in the
  plugin (hip-kernel-provider has two).
- **Pass the engine's `--test-config`**, as the lane does. Cases the TOML skips
  are skipped before observation, so you do not claim what the lane never runs.
- A `--gtest_filter` narrows the run; claims for filtered-out graphs are left as
  they were.
- Run one writer at a time per bundle tree; concurrent writers race.

What it does: for each graph it asks the engine for its ranked engine list,
records the engine **as supporting the graph if it is ranked**, and skips the
test (`support-claim authoring run (--write-support-claims)`). It does not
execute or compare anything. It only ever **adds** claims for the running
machine's base arch token and platform, merges them into existing sidecars,
creates a sidecar only when there is something to claim, and writes canonical
JSON — a run that changes nothing leaves no git diff.

It prints a summary instead of the claim summary:

```text
==== SUPPORT CLAIM WRITE SUMMARY ====
  graphs registered: N  observed: N  not observed: N  skipped in SetUp: N  unaccounted for: N
  observations: N  written: N  unchanged: N  skipped: N  errors: N
```

It exits 1 when nothing was observed at all, when any sidecar could not be
written, when a graph reached the engine but could not be observed (it failed to
open, or the query did not resolve), or when registered graphs are unaccounted
for without a `--gtest_filter` or shard split to explain them. Graphs removed by
a filter, or skipped in `SetUp()`, are reported and their claims left as they
were; they do not fail the run.

### Confirm, review, commit

1. **Confirm with an enforcing run** of the same lane (no `--write-support-claims`).
   The new cells should appear as `confirmed` and not under `failed_in_use`.
   The writer only saw that the engine *accepts* the graph; the enforcing run
   is what shows it gets it right. Retract any new claim that lands in
   `failed_in_use` before committing.
2. **Review the diff.** Expect additions only. The writer never removes a claim.
3. **Commit the sidecars.** The `verify-support-claims` pre-commit hook checks
   schema, canonical form, sweep case ids, and orphans. Run it directly with
   `python dnn-providers/integration-tests/scripts/verify_support_claims.py`.

Each machine can only claim its own cell. Claims for other archs or platforms
come from runs on those machines.

### Retract a claim

Retraction is always a deliberate, reviewed edit — no tool removes a claim,
because a regression and an intended withdrawal look identical to the harness.
Delete the engine's arch/platform (or move the sweep case out of its group) by
hand, keep the file in canonical form (2-space indent, sorted keys, trailing
newline; the verifier reports `not in canonical form` otherwise), and say in the
PR why the engine no longer supports it.

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

Do not use a skip to hide a regression on a claimed cell; the claim is there to
make that regression visible.

For arch-wide exclusions of whole suites, the tier YAML's `exclude_gpu` block is
the coarser tool; a `"*"` pattern disables the suite on that arch entirely and
should carry a comment and a tracking issue.

## C++ tests: the CMake rule

Every `.cpp` under `src/integration-tests/` must be registered through exactly
one of two functions, or configure fails listing it as an orphan:

| Function | Builds | Use for |
|---|---|---|
| `add_cpp_graph_test_sources(...)` | only with `-DBUILD_CPP_GRAPH_TESTS=ON` | legacy C++ graph tests |
| `add_always_built_test_sources(...)` | always | tests that cannot be a bundle |

`add_always_built_test_sources()` also requires the file to be listed in
`HIPDNN_IT_ALWAYS_BUILT_SOURCES` at the top of
`src/integration-tests/CMakeLists.txt`, with a comment saying why it cannot be a
bundle. That list is the one central, reviewable place a new C++ test can enter
CI.

C++ test names are checked by the `*_test_name_validation` CTest entries: a test
**case** name must not contain the suite keywords `Test`, `Integration`, `Gpu`,
or a datatype token such as `Fp16`.

## Reference-executor tests

`tests/` holds the tests for the GPU/CPU reference executors themselves
(`hipdnn_gpu_ref_tests`), not for any engine. A new parameterized suite there
defines all four tiers and names cases by shape tag:

```cpp
INSTANTIATE_TEST_SUITE_P(Smoke,         MyNewOp2dTestFp32, ::testing::ValuesIn(getSmallCases()),      byTag());
INSTANTIATE_TEST_SUITE_P(Standard,      MyNewOp2dTestFp32, ::testing::ValuesIn(getMediumCases()),     byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive, MyNewOp2dTestFp32, ::testing::ValuesIn(getLargeEdgeCases()),  byTag());
INSTANTIATE_TEST_SUITE_P(Full,          MyNewOp2dTestFp32, ::testing::ValuesIn(getLargeStressCases()), byTag());
```

Follow the shape-catalog pattern in `tests/gpu-ref/ConvShapeCatalog.hpp`; new
shapes added to an existing catalog are picked up automatically. Some existing
files use `Quick` rather than `Smoke` for the first tier — match the file you
extend.

## Before you open the PR

- [ ] New graph coverage is a bundle, not a C++ test.
- [ ] The lane for each engine you touched ran, and you read its
      `TEST COVERAGE SUMMARY` and `SUPPORT CLAIM SUMMARY` — not just the exit code.
- [ ] `unclaimed_support` for your new bundles is recorded in sidecars, and none
      of the new claims is in `failed_in_use`.
- [ ] Golden data, if any, is `dvc commit`ted and `dvc push`ed; only `.json` and
      `.dvc` files are in git.
- [ ] `verify-support-claims` passes (it runs in pre-commit).
- [ ] If you regenerated or renamed sweep cases, the `ffm-quick` ids in each
      provider's `test_categories_integration.yaml` still exist
      (`ctest -L ffm-quick -N`), and no sidecar names a case that no longer exists.
