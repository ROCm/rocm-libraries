# Adding Tests and Updating Claims

This page shows how to add coverage to the cross-provider suite. It also shows
how to keep the metadata, support claims, golden data and engine config
consistent with that coverage. The [index](README.md#terms) defines terms such
as *cell*, *lane* and *oracle*.

- [File Formats](file-formats.md) describes the files that this page changes.
- [Running the Tests](running-tests.md) shows how to run what you added and
  how to read the result.

Run commands from the repository root. The superbuild is in `build/`.

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

- **Sweep** — the same topology across several shapes, dtypes or layouts. This
  is the common case. A new shape is one more case, not a new directory.
- **Single graph** — exactly one concrete graph with nothing to vary, a graph
  that you want to keep byte-for-byte, or cases whose *structure* differs (a
  sweep can vary values, not node count or wiring).

If you are not sure, `import_graph.py` decides. It appends to a sweep whose
skeleton matches. If no sweep matches, it creates a new one.

A bundle of any kind is complete only when it has all three required parts
([formats](file-formats.md#map)):

1. **The graph** — `{Name}.json`, or `graph.template.json` plus a `sweep.json` case.
2. **Metadata** — `{Name}.meta.json`, or `cases[].metadata` on every sweep case,
   with `format_version`, `generator` and `reference_source`.
3. **A support-claim sidecar** — `{Name}.support.json`, or the sweep's
   `support.json` — even if it claims nothing yet.

Golden data is the only optional part.

## End to end: a bundle for a new op

This is the whole path, in order. Each step links to its details.

1. **Get graph JSON.** Capture it from an existing C++ graph test. If the op
   has no test, first write a parameterized C++ graph test, then capture it.
   To take a graph from any running application, set `HIPDNN_LOG_GRAPH_DIR` to
   a directory. Each graph that is finalized is written there as
   `graph_<id>.json`
   ([details](../../../projects/hipdnn/docs/Environment.md#hipdnn_log_graph_dir)).
   See [A brand-new sweep](#a-brand-new-sweep).
2. **Import it** with `import_graph.py`
   ([details](#from-a-graph-you-already-have)). This writes the graph, the case
   metadata and an empty `support.json`, and prints the GTest name.
3. **Check the oracle.** If no reference executor supports the op, decide
   between golden data and a lower `enforcement_level`
   ([details](#when-no-reference-executor-supports-the-op)).
4. **Re-run CMake**, so the build's copy of the bundle tree gets the new
   files ([why](running-tests.md#the-build-reads-a-copy-of-the-bundles)).
5. **Run the lane** for each engine that you can run. Make the output visible
   (`ctest ... -V`, or the check target). Read both summaries.
6. **Record claims** for the engines that accept the graph
   ([details](#record-claims-with---write-support-claims)). Then re-run CMake
   and the lane to confirm the claims.
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

The tool hashes the skeleton of the graph. Then it does one of these:

1. **Exact duplicate** (same graph, seed and inputs) → prints `DUPLICATE` and
   writes nothing. `--strict` makes that exit non-zero. `--force` appends anyway.
2. **New case for an existing topology** → appends one case to that sweep's
   `sweep.json` and leaves every existing case untouched. If no matching sweep
   is in `--tier`, it appends to a match in another tier. The printed path and
   GTest name show where the case went.
3. **New topology** → creates a new sweep directory under `{tier}/{Op}/`. It
   never writes into a directory that already holds a sweep. If the preferred
   name is taken, it uses the next free `Variant2`, `Variant3`, ….

In all cases the tool:

- prints the case id and the GTest name that the harness registers
  (`gtest name: quick_Relu_Nchw.2_3_4_5_fp32_nchw`);
- sets `generator` to `import_graph.py`, unless the capture sidecar or
  `--meta generator=…` sets one;
- warns if there is no `reference_source`;
- creates an empty `support.json` in the sweep directory if it has none. It
  never changes an existing `support.json`.

Other flags: `--seed`, `--meta key=value` (repeatable), `--dry-run`. `seed`,
`format_version`, `minimum_vram_mb` and `inputs` parse as JSON, so
`--meta seed=42` is the number 42; NaN and Infinity are refused. Any other key
is stored as the string given.

Afterwards, review `git status` and `git diff`. Expect one new case, or one new
directory with three files.

### From an existing C++ graph test

Capture its graphs, then import each one. Capture needs a build with
`-DBUILD_CPP_GRAPH_TESTS=ON`. Without it, the C++ graph tests are not
compiled.

```bash
build/bin/hipdnn_integration_tests --capture-bundles /tmp/captured \
    --gtest_filter='Full/IntegrationGpuConvFwdBiasActiv2dFp16.Correctness/*'

find /tmp/captured -name '*.json' ! -name '*.meta.json' | while read -r graph; do
    python3 dnn-providers/integration-tests/migration-scripts/import_graph.py \
        --graph "$graph" \
        --bundle-dir dnn-providers/integration-tests/integration-test-bundles
done
```

Capture writes `{suite}/{case}/{case}.json` and a `.meta.json` beside each
graph. The suite name keeps its `/` (`Full/IntegrationGpu…`). The files are
therefore one directory deeper than expected, so the example uses `find`.

`import_graph.py` reads the `.meta.json` beside the graph and uses it as the
case metadata, as `place_bundles.py` does, so no `--seed`, `--meta` or `--tier`
flags are needed:

- the seed and `inputs` fill specs carry over, so the case reproduces the
  inputs of the C++ test;
- `reference_source` is `c++ integration suite: <Suite>.<Case>`, derived from
  the sidecar's `operation` and the capture path;
- the suite prefix picks the tier: `Smoke/` → `quick`, `Full/` → `full`,
  `Standard/` → `standard`, `Comprehensive/` → `comprehensive`, anything else
  → `quick`.

A `--seed`, `--meta` or `--tier` value that contradicts the sidecar is applied
but prints a `WARN` naming both values; `--strict` makes it an error. A sidecar
that exists but is unusable (bad JSON, not an object, an `inputs` key that is
not a tensor UID, an `operation` that is not a non-empty string) fails the
import, with or without `--strict`. A graph with no sidecar lands in `--tier`
(default `quick`). After each printed test runs and passes as a bundle, delete
the C++ registration.
[`migration-scripts/README.md`](../migration-scripts/README.md) describes the
bulk pipeline and its byte-level verification.

### A brand-new sweep

No tool generates the case matrix of a sweep from a list of shapes. Cases are
not written by hand. Use a round trip through C++:

1. Write the matrix as a parameterized C++ graph test in
   `src/integration-tests/<op>/`. Follow an existing test (for example
   `src/integration-tests/rmsnorm/IntegrationGpuRMSNorm.cpp`). Register it
   with `add_cpp_graph_test_sources()` in the `CMakeLists.txt` of that
   directory.
2. Build with `-DBUILD_CPP_GRAPH_TESTS=ON`. Run the test until it passes.
3. Capture and import it as above.
4. Delete the C++ test once the bundle cases run and pass.

### When no reference executor supports the op

The harness checks the output of an engine only if an oracle exists
([Verification modes](running-tests.md#verification-modes)).
`src/harness/reference-validation/ReferenceOpCoverage.hpp` lists the ops that
each reference executor must handle. If your op is not in that list, the bundle
**fails** with an `Unverifiable: … tried: …` message, and it appears in
`UNVERIFIABLE BUNDLES`: an engine whose output nothing checks is untested. Do
one of these:

- **add golden data** produced by an independent implementation (see
  [Golden data with DVC](#golden-data-with-dvc)), or
- **add the op to a reference executor**, or
- **set `enforcement_level`** in the metadata to `applicability` (the engine
  accepts the graph) or `buildable` (its plans compile), so that the bundle
  claims only what the harness can check today, or
- **quarantine the bundle** until one of the above lands: exclude it in the
  provider's `test_categories_integration.yaml` with a comment that says why.

### Finding existing cases

```bash
python3 dnn-providers/integration-tests/migration-scripts/find_case.py --op Batchnorm
python3 dnn-providers/integration-tests/migration-scripts/find_case.py --id f446b9 --detail   # includes the --gtest_filter
```

[`migration-scripts/README.md`](../migration-scripts/README.md) describes more
queries (`--dtype`, `--layout`, `--input`, `--shape`).

### Choosing a tier

Put the bundle in the cheapest tier that still finds the failures that the
bundle is meant to find. `quick` runs in the PR tier of every provider lane, so
put large shapes in `standard` or higher. The directory *is* the tier.
[Test tiers](running-tests.md#test-tiers) shows what each tier runs.

## Golden data with DVC

### Golden data is optional

A bundle with no golden data is still a full test: the harness compares its
output with the GPU or CPU reference executor. Add golden data if you want the
stricter golden comparison, or if no reference executor can run the op.

Run DVC commands from the repository root. Reads are anonymous. `dvc push`
needs AWS credentials. After any pull, re-run CMake so that the build's copy
has the data.

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

`dvc push` uploads from the local **cache**, not the working tree. Always run
`dvc commit` first. If you do not, the push silently skips the new files.
Legacy ops whose pointers have no `remote:` key use the default `storage`
remote (`dvc push` with no `-r`).

### Add golden data to a sweep case

Put the files in `golden/{CaseId}/` (`tensor<uid>.bin` and a `tensors.dvc`
pointer written as above). Then add `"golden": {"path":
"golden/{CaseId}/tensors.dvc"}` to that case in `sweep.json`. You can edit the
fields of an existing case by hand.

### Update or remove golden data

- **Update:** overwrite the `.bin` files. If the set of files changed,
  re-author the `outs:` list of the pointer first. Run `dvc commit -f` on the
  pointer, commit the pointer, then run `dvc push`.
- **Remove a bundle:** run `dvc remove` on its pointer, delete the directory,
  and commit.
- **Roll back:** revert the pointer in git. Then `dvc pull` gets the old data.
  The old data stays in S3 by content hash.

### Generated bundles

A generator, not a capture, produces some ops. Examples are
`integration-test-bundles/quick/SdpaFwd/generate_golden_data.sh` and the
generators in `reference-data-scripts/`. Regenerate, then run:

```bash
dvc commit -f -R <op dirs>
dvc push -r golden-data -R <op dirs>
git add <op dirs>          # .json, .meta.json, .support.json, .tensors.dvc — .bin is ignored
```

`dvc commit` keeps the existing `remote:` key of each pointer. If a generator
does not write a support sidecar, you must write it.

### Check the data itself

```bash
python dnn-providers/integration-tests/reference-data-scripts/verify_golden_bundles.py \
    --require-data <each bundle or sweep directory you added or changed>
build/bin/hipdnn_golden_data_tests --reference cpu
```

The first command checks graph JSON, metadata (including the required
`generator` and `reference_source`), tensor sizes, and NaN/Inf in outputs.
`--require-data` makes a pointer whose `.bin` was not pulled an error instead
of a warning. The second command recomputes each golden bundle with a reference
executor and compares the results.

Pass the directories that you changed. Do not pass the whole
`integration-test-bundles/` tree, because the tree does not pass today. There
are two reasons:

- The verifier expects exactly `{tier}/{Op}/{Layout}/{DataType}/{Name}/{Name}.json`.
  The SDPA single-graph bundles either carry an extra variant directory
  (`…/hd128_nomask_batch/Gqa/Gqa.json`) or sit one level up
  (`…/hd128_causal_mha/Prefill.json`). The verifier reports them as
  `graph files must be named <BundleName>/<BundleName>.json`.
- Many older sweeps lack `generator` or `reference_source` in their case
  metadata.

Your bundles must add no errors of their own.

### DVC troubleshooting

| Symptom | Fix |
|---|---|
| `dvc push` auth error | Run `aws sts get-caller-identity`. Writes need AWS credentials. Reads do not. |
| A `.dvc` pointer exists but no `.bin` on disk | `dvc pull path/to/Name.tensors.dvc` |
| A `.bin` was committed to git by accident | Run `git rm --cached` on it, then run `dvc commit -f` on its pointer. |
| Tensor files were added or removed | Re-author the `outs:` list of the pointer, then run `dvc commit -f` on it. |
| Tests do not see new data | Re-run CMake. Then run `dvc status` to check for drift between pointers and the cache. |

## Updating support claims

Support-claim sidecars ([format](file-formats.md#support-claim-sidecars--namesupportjson-and-supportjson))
record which engines accept which graphs on which arch and platform. They are
**written by the harness, reviewed by people**.

### When to update them

| Trigger | Action |
|---|---|
| A run's `SUPPORT CLAIM SUMMARY` lists entries under **`unclaimed_support`** | The engine accepts graphs that nobody has claimed. Record them. This is the main trigger. It never fails a run, so look for it. |
| You added bundles | Record claims for the engines and archs that you can run. If no engine accepts the graph, keep the [empty sidecar](#when-no-engine-accepts-the-graph-yet). |
| `verdicts` all zero while `unenforced.no_applicable_claim` > 0 | This arch/platform has no claims yet (typical on a bring-up ASIC). Record them. |
| `claim_failures` → `CLAIM_BROKEN`, and dropping the support is **intended** | [Retract the claim](#retract-a-claim). Otherwise fix the engine. |
| `failed_in_use` | A claimed cell the engine gets wrong. Fix the engine, or retract the claim until it is fixed. |

Only bundles that already have a sidecar can report `unclaimed_support`. A
bundle with no sidecar reports nothing.

An `unclaimed_support` entry names the bundle and the sweep cases. The `run`
block of the summary names the engine, arch and platform where the entry was seen:

```json
"run": { "arch": "gfx942", "engine": "MIOPEN_ENGINE", "platform": "linux" },
"unclaimed_support": [
  { "bundle": "integration-test-bundles/quick/ConvFwd/sweep.json",
    "cases": ["case_a", "case_b"], "reached": "verified", "required": "verified" }
]
```

Check `reached` before you claim. If `reached` equals `required`, the run
verified the result. If `reached` is below `required`, the engine accepts the
graph but nothing checked its output. The writer below claims it anyway, so
decide first if you want it claimed.

### Record claims with `--write-support-claims`

Run this on a machine with the arch and platform that you want to claim:

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
  place. The next configure overwrites the copy in the build.
- **Pass `--test-engine`.** Without it, the writer records every engine in the
  plugin (hip-kernel-provider has two).
- **Pass the `--test-config` of the engine**, as the lane does. Cases that the
  TOML skips do not reach observation, so you do not claim what the lane never
  runs.
- **Filter to bundles** (as above, or narrower). The writer observes bundles
  only. Any C++ tests in the binary still run normally and can fail the run.
  Graphs that a filter removes keep their claims unchanged.
- Run only one writer at a time for each bundle tree. Concurrent writers cause
  a race condition.

For each bundle, the writer asks for the ranked list. It records the engine
**as supporting the graph if it is ranked**. Then it skips the test
(`support-claim
authoring run (--write-support-claims)`). It does not execute or compare the
bundle. It only **adds** claims, for the arch token and platform of the running
machine. It merges the claims into existing sidecars and writes them in the
normal form of the writer
([details](file-formats.md#support-claim-sidecars--namesupportjson-and-supportjson)).
It leaves no git diff when nothing changed. It does not create a sidecar for a
graph that no engine accepted.

It prints a summary instead of the claim summary:

```text
==== SUPPORT CLAIM WRITE SUMMARY ====
  graphs registered: N  observed: N  not observed: N  skipped in SetUp: N  unaccounted for: N
  observations: N  written: N  unchanged: N  skipped: N  errors: N
```

The writer exits 1 if any of these is true:

- Any test in the run failed.
- Zero tests were selected.
- Nothing was observed at all.
- A sidecar could not be written.
- A graph reached the engine but could not be observed (it failed to open, or
  the query did not resolve).
- Registered graphs are unaccounted for, and no `--gtest_filter` or shard split
  explains them.

The writer reports graphs that a filter removed or that were skipped in
`SetUp()`. Their claims stay unchanged.

### Confirm, review, commit

1. **Re-run CMake**, so the lane reads the sidecars that you just wrote.
   Alternatively, run the binary with `--golden-data-dir` set to the source tree.
2. **Run the lane enforcing** (no `--write-support-claims`) and read the claim
   summary. Expect these results: the new cells no longer appear under
   `unclaimed_support`, none is under `failed_in_use`, and `verdicts.confirmed`
   has grown. `confirmed` and `accepted` are counts only. A claimed cell that
   stayed `accepted` was never verified. Find it under `UNVERIFIABLE BUNDLES`.
3. **Review the diff.** Expect additions only. The writer never removes a claim.
4. **Verify and commit:**

   ```bash
   python dnn-providers/integration-tests/scripts/verify_support_claims.py
   ```

   Run it by hand. The pre-commit configuration currently hides
   `integration-test-bundles/` from every hook, including this one.

Each machine can only claim its own cell. Claims for other archs or platforms
come from runs on those machines.

### Harvest claims from CI logs

Instead of running `--write-support-claims` on each machine, you can take the
support that a CI run proved and add it to the sidecars. A run of
`hipdnn_integration_tests` prints a `SUPPORT CLAIM SUMMARY` whose
`unclaimed_support` names graphs that reached the depth their bundle requires but
that no sidecar claims. `scripts/harvest_support_claims.py` reads these blocks
from CI job logs. Run it from `dnn-providers/integration-tests/`. It reads first,
and only `--write` changes files:

```bash
python3 scripts/harvest_support_claims.py                                                # report every arch/platform of the latest develop nightly
python3 scripts/harvest_support_claims.py --run <id> --arch gfx942 --platform linux      # report one; name the sidecars that would change
python3 scripts/harvest_support_claims.py --run <id> --arch gfx942 --platform linux --write
```

- **Sources.** `--run <id>` fetches the logs of one workflow run with the `gh`
  CLI. A positional argument is a log file, or a directory of `*.log` files, on
  disk. With neither, the latest completed scheduled develop nightly is used.
  `--jobs <regex>` picks the jobs (default: every provider's test job).
  `--min-runs N` claims a cell only when it was kept in at least N runs.
- **A cell is kept only if its gtest printed an `OK` line.** `reached: verified`
  means that the comparison ran, not that it passed. A cell that FAILED, was
  SKIPPED, or reached less than its required depth is dropped. A drop in any run
  vetoes the cell. A block with bad JSON, an inconsistent counter or a harness
  defect rejects its whole log, and then nothing is written (exit 1).
- **`--write` needs both `--arch` and `--platform`**, so that one change covers
  one arch/platform. `--arch` takes one gfx target, not a family name such as
  `gfx94X`.
- **Add-only, and the sidecars must be current.** The script fetches the tip of
  develop. It refuses a sidecar that lacks a claim its develop version has:
  rebase onto develop first. It also refuses a sidecar that does not render back
  to its own bytes. If any sidecar or log is refused, it writes nothing.
- `claim_failures` and `failed_in_use` are listed for you to read. They are never
  written as claims.

Then continue with [Confirm, review, commit](#confirm-review-commit).

### When no engine accepts the graph yet

A bundle still needs a sidecar if no engine that you can run accepts it.
Examples are a new op whose engine support has not landed, or a graph for
hardware that you do not have. `import_graph.py` creates this empty sidecar for
you. Otherwise, commit it by hand, exactly as shown (include the trailing
newline):

```json
{
  "claims": {},
  "version": 1
}
```

It records that the bundle was reviewed and nothing is claimed yet. After an
engine accepts the graph, a run reports it under `unclaimed_support`. Then
`--write-support-claims` adds the claims.

### Retract a claim

Retraction is always a deliberate, reviewed edit. No tool removes a claim,
because a regression and an intended withdrawal look the same to the harness.

1. Delete the arch/platform entry of the engine by hand (or move the sweep case
   out of its group). Keep the file in the normal form of the writer.
2. If the engine still *accepts* the graph (it is broken in use, not declined),
   also add a `[[test_skips]]` entry for that case and arch to the TOML of the
   engine. If you do not, the next `--write-support-claims` run observes the
   graph as supported and adds the claim again immediately.
3. In the PR, say why the engine no longer supports the graph. Link the
   tracking issue.

## Record an engine's known limitation

Edit the engine's `config/<ENGINE_NAME>.toml`
([format](file-formats.md#per-engine-test-config--configengine_nametoml)):

- **The engine has no kernel for a case on some arch** → `[[test_skips]]` with
  `archs`/`platforms` as narrow as the evidence supports, and a `reason` that
  links the tracking issue. Every run shows the skip as `[arch …] <reason>`.
- **The engine's numerics legitimately differ** → `[[tolerance_overrides]]`
  for the affected tests only.
- **Per-element comparison is not suitable for an output** (long
  reductions whose elements cancel toward zero) → `[[validator_overrides]]`
  with `validator = "rms"` for those tensors only.
- **An output is correctly infinite on both sides** (a fully masked SDPA row's
  log-sum-exp) → `[[validator_overrides]]` with
  `validator = "allclose_matching_infinities"` for those tensors only. It runs
  on the host only. The comparison must run on the host (`--validator cpu` or
  `HIPDNN_TEST_VALIDATOR=cpu`). If it does not, the tensor fails.

Do not use a skip to hide a regression on a claimed cell, unless you also
retract the claim. The claim makes that regression visible.

To exclude whole suites for an arch, use the `exclude_gpu` block of the tier
YAML. It is the coarser option. A `"*"` pattern disables the suite on that arch
entirely. Add a comment and a tracking issue to it.

## Add an engine lane

To add a new engine to the suite, register a lane in the `CMakeLists.txt` of
its provider. The suite installs a CMake package, `hipdnn_integration_tests`.
The package exports the binary and `cmake/HipdnnIntegrationTestHelpers.cmake`:

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

This code creates the check target. From the YAML, it also creates one CTest
suite for each tier. Each suite runs
`hipdnn_integration_tests --test-article <plugin> --test-engine <ENGINE> --test-config <toml> --gtest_filter=<tier patterns>`.
Every lane names `--test-engine`, so every lane enforces support claims. The
function also accepts `REFERENCE_EXECUTOR`, `GTEST_FILTER`, `ENVIRONMENT`,
`ENVIRONMENT_MODIFICATION`, `FIXTURES_REQUIRED`, `INSTALL_TEST_FILE` and
`INSTALL_ENVIRONMENT`. `GTEST_FILTER` applies to the check target and to a lane
without a tier YAML. It does not apply to the tier suites. If you omit
`TEST_CATEGORIES_YAML`, the lane is one CTest test with no tier labels. If the
package is not available, the lane is skipped at configure time.

A new lane also needs its engine TOML and tier YAML
([formats](file-formats.md#per-engine-test-config--configengine_nametoml)).
Copy the files of an existing provider and keep the same tier shape.

## C++ tests: the CMake rule

You must register every `.cpp` under `src/integration-tests/` with exactly one
of two functions. If you do not, configure fails and lists the file as an orphan:

| Function | Builds | Use for |
|---|---|---|
| `add_cpp_graph_test_sources(...)` | only with `-DBUILD_CPP_GRAPH_TESTS=ON` | C++ graph tests |
| `add_always_built_test_sources(...)` | always | tests that cannot be a bundle |

`add_always_built_test_sources()` also requires the file in
`HIPDNN_IT_ALWAYS_BUILT_SOURCES` at the top of
`src/integration-tests/CMakeLists.txt`. Add a comment that says why the test
cannot be a bundle. That list is the one central, reviewable place where a new
C++ test can enter CI.

To run in the PR tiers of the provider lanes, a C++ test must be instantiated
with a `Smoke`, `Quick` or `Standard` prefix. A test with no tier prefix runs
only in `comprehensive` and higher tiers
([Test tiers](running-tests.md#test-tiers)).

Put tests of behavior that is specific to one plugin in the
`integration_tests/` directory of that provider (for example
`dnn-providers/miopen-provider/integration_tests/`). Do not put them in the
shared suite.

The `*_test_name_validation` CTest entries check C++ test names. A test **case**
name must not contain the suite keywords `Test`, `Integration`, `Gpu`, or a
datatype token such as `Fp16`.

## Reference-executor tests

`tests/gpu-ref/` contains the tests for the GPU/CPU reference executors
themselves (`hipdnn_gpu_ref_tests`). It does not contain tests for any engine.
To add a test:

1. Add the `.cpp` to the `add_executable(hipdnn_gpu_ref_tests …)` list in
   `tests/CMakeLists.txt`. A file that is not in the list is never compiled.
2. Follow the shape-catalog pattern in `tests/gpu-ref/ConvShapeCatalog.hpp`.
3. Define all four tiers and name cases by shape tag. The case-list functions
   below are placeholders for the functions of your catalog:

   ```cpp
   INSTANTIATE_TEST_SUITE_P(Smoke,         MyNewOp2dTestFp32, ::testing::ValuesIn(getSmallCases()),      byTag());
   INSTANTIATE_TEST_SUITE_P(Standard,      MyNewOp2dTestFp32, ::testing::ValuesIn(getMediumCases()),     byTag());
   INSTANTIATE_TEST_SUITE_P(Comprehensive, MyNewOp2dTestFp32, ::testing::ValuesIn(getLargeEdgeCases()),  byTag());
   INSTANTIATE_TEST_SUITE_P(Full,          MyNewOp2dTestFp32, ::testing::ValuesIn(getLargeStressCases()), byTag());
   ```

The harness finds new shapes in an existing catalog automatically. Some
existing files use `Quick` instead of `Smoke` for the first tier. Match the file
that you extend. The other files directly under `tests/` are the unit tests of
the harness (`hipdnn_integration_tests_unit_tests`).

## Before you open the PR

- [ ] New graph coverage is a bundle, not a C++ test.
- [ ] Every new bundle has its graph, its metadata (with `generator` and
      `reference_source`) and a support sidecar, even an empty one. No tool
      checks for a missing sidecar, so check by hand.
- [ ] `verify_golden_bundles.py` reports no errors for the bundle directories you
      added or changed ([Check the data itself](#check-the-data-itself)).
- [ ] You re-ran CMake after you changed bundles or sidecars or pulled data.
      You ran the lane for each engine that you changed, with its output
      visible. You read its `TEST COVERAGE SUMMARY` and `SUPPORT CLAIM SUMMARY`,
      not only the exit code.
- [ ] `unclaimed_support` for your new bundles is recorded (or deliberately left
      unclaimed), and no new claim is in `failed_in_use`.
- [ ] If there is golden data, you ran `dvc commit` and `dvc push` for it. Only
      `.json` and `.dvc` files are in git.
- [ ] `python dnn-providers/integration-tests/scripts/verify_support_claims.py`
      passes when you run it by hand.
- [ ] If you regenerated or renamed sweep cases, no sidecar names a case that no
      longer exists. The `ffm-quick` ids of miopen-provider still exist
      (`ctest --test-dir build/dnn-providers/miopen-provider -L ffm-quick -N`).
