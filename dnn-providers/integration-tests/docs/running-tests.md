# Running the Integration Tests

How to run the cross-provider suite against an engine, how the tiers and
filters decide what runs, and how to read what it prints — including the ways a
run can look green while testing nothing.

- File formats named here are described in [File Formats](file-formats.md).
- Acting on what a run reports (new bundles, stale claims) is in
  [Adding Tests and Updating Claims](adding-tests.md).

## The binaries

| Binary | Tests | Engine loaded |
|---|---|---|
| `hipdnn_integration_tests` | Bundles and sweeps (plus any C++ tests compiled in) against **one provider's engine** | yes |
| `hipdnn_golden_data_tests` | Our checked-in golden `.bin` data against the CPU/GPU reference executors | **no** |
| `hipdnn_gpu_ref_tests` | The GPU/CPU reference executors themselves | no |
| `hipdnn_integration_tests_unit_tests` | The harness (discovery, TOML, claims, verdicts) — no device needed | no |

`hipdnn_integration_tests` is built once in
`dnn-providers/integration-tests/` and run by **each provider** against its own
plugin. That is how one bundle tree validates every engine.

## Three ways to run it

### 1. CTest by tier (what CI does)

Each provider registers one CTest suite per tier from its
`test_categories_integration.yaml`. The suites live in the **provider's** build
directory:

```bash
cd build/dnn-providers/miopen-provider
ctest -L quick --output-on-failure
ctest -L quick -N                      # list what would run, without running it
```

From the superbuild root this only works when the build was configured with
`-DROCM_LIBS_ENABLE_ROOT_CTEST=ON` (CI sets it; the default is off). Without it,
`ctest --test-dir build` finds nothing — and exits 0.

Every external-integration suite carries the labels `integration_test`,
`external_integration_test`, `slow` and its engine name, plus the tier labels
from the YAML. Two `-L` flags must both match:

```bash
ctest -L quick -L MIOPEN_ENGINE        # quick tier, MIOpen lane only
```

Suite names are `<prefix>_<category>_suite`, where the provider sets the prefix —
for example `miopen-provider-external-integration_quick_suite`.

### 2. The provider's check target (whole suite, no tier filter)

```bash
cmake --build build --target miopen-provider-external-integration-check
```

This runs every registered bundle for that engine, unfiltered. It is the right
way to see an engine's full picture and the wrong way to reproduce a tier.

| Provider | Target | Engine |
|---|---|---|
| miopen-provider | `miopen-provider-external-integration-check` | `MIOPEN_ENGINE` |
| hipblaslt-provider | `hipblaslt-provider-external-integration-check` | `HIPBLASLT_ENGINE` |
| hip-kernel-provider | `hip-kernel-provider-external-integration-check` | `HIP_MLOPS_ENGINE` |
| hip-kernel-provider | `hip-kernel-provider-asm-sdpa-external-integration-check` | `ASM_SDPA_ENGINE` |

hip-kernel-provider exposes two engines and registers one target per engine.
The authoritative list is the set of `add_external_integration_test_target()`
calls under `dnn-providers/`.

#### How a provider registers its lane

The suite installs a CMake package, `hipdnn_integration_tests`, exporting the
binary and `cmake/HipdnnIntegrationTestHelpers.cmake`. A provider registers one
lane per engine:

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

That creates the check target above and, from the YAML, one CTest suite per
tier running
`hipdnn_integration_tests --test-article <plugin> --test-engine <ENGINE> --test-config <toml> --gtest_filter=<tier patterns>`.
Because every lane names `--test-engine`, every lane enforces support claims.
The function also accepts `REFERENCE_EXECUTOR`, `GTEST_FILTER`, `ENVIRONMENT`,
`ENVIRONMENT_MODIFICATION`, `FIXTURES_REQUIRED`, `INSTALL_TEST_FILE` and
`INSTALL_ENVIRONMENT`. Without the package the lane is skipped at configure time.

### 3. The binary directly (debugging one case)

Pass the plugin, pin the engine, and pass that engine's TOML — exactly what the
CTest lanes do:

```bash
./bin/hipdnn_integration_tests \
    --test-article /path/to/libmiopen_plugin.so \
    --test-engine  MIOPEN_ENGINE \
    --test-config  dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml \
    --gtest_filter='quick_RMSNorm_Default.*'
```

- Always pass `--test-engine` when you are testing one provider. Without it,
  hipDNN's normal engine selection picks the winner, a "pass" may have come from
  a different engine, and support claims are not enforced.
- On Windows the binary needs the ROCm `bin` directory and the build's `bin`
  directory on `PATH`; without them it dies with `STATUS_DLL_NOT_FOUND`
  (`0xc0000135`) before running anything.
- Standard GTest sharding (`GTEST_TOTAL_SHARDS` / `GTEST_SHARD_INDEX`) works.

## Command-line reference — `hipdnn_integration_tests`

| Flag | Env var | Default | Purpose |
|---|---|---|---|
| `--test-article`, `--ta` | | plugin discovery | Path to the engine plugin (`.so` / `.dll`). Only that plugin is loaded. |
| `--test-engine`, `--te` | | none | Pin the run to one engine. An engine that is not loaded exits 1 before any test: `Error: Engine '<name>' is not loaded. Check the plugin path.` |
| `--test-config`, `--tc` | | none | The engine's TOML ([format](file-formats.md#per-engine-test-config--configengine_nametoml)). A missing path or invalid TOML exits 1. |
| `--verification-mode`, `--vm` | `HIPDNN_TEST_VERIFICATION_MODE` | `auto` | How bundle output is checked; see [Verification modes](#verification-modes). |
| `--validator` | `HIPDNN_TEST_VALIDATOR` | `auto` | Where the comparison runs: `auto` (follow the reference), `cpu`, `gpu`. |
| `--reference-executor` | `HIPDNN_TEST_REFERENCE_EXECUTOR` | `cpu` | Reference for C++ graph tests only. |
| `--golden-data-dir`, `--gd` | `HIPDNN_TEST_GOLDEN_DATA_DIR` | `<exe>/../lib/integration-test-bundles/` | Bundle root. Must exist. |
| `--no-bundles` | `HIPDNN_TEST_ALLOW_BUNDLES=0` | bundles on | Register only the C++ tests compiled into the binary. |
| `--enforce-support-claims[=true\|false]` | | on | Fail a test whose sidecar claim the engine breaks. See [Support claims](#support-claim-summary). |
| `--write-support-claims` | | off | Authoring run: record which graphs the engine accepts into sidecars, run nothing else. See [Adding Tests](adding-tests.md#updating-support-claims). |
| `--skip-graph-validation` | | off | Pass as soon as the engine accepts the graph; do not execute or compare. |
| `--fail-on-unsupported` | | off | Fail instead of skip when no engine supports a graph. **C++ graph tests only**; bundles ignore it. |
| `--generate-support-matrix[=file]` | | off | Write a markdown matrix. **C++ graph tests only**; on a default build it writes an empty table. |
| `--capture-bundles <dir>` | | off | Dump compiled-in C++ graph tests as bundle JSON (migration tooling). |
| `--gtest_*` | | | Passed through to GTest. |

Flag combinations the binary refuses (exit 1):

- `--enforce-support-claims` typed without `--test-engine`.
- `--write-support-claims` together with an explicit `--enforce-support-claims`.
- `--write-support-claims` without `--test-article`, or without a bundle root
  (`--golden-data-dir` or `HIPDNN_TEST_GOLDEN_DATA_DIR`).

When enforcement is merely inherited (not typed) and there is no
`--test-engine`, the run quietly falls back to reporting claims without
enforcing them.

A machine with no HIP device prints `No HIP devices available; skipping …` and
exits 0 immediately.

## `hipdnn_golden_data_tests`

Recomputes every bundle that has golden data with a reference executor and
compares it against the checked-in `.bin` files. No plugin, no engine, no claims:
it validates **our data**, not a provider.

```bash
./bin/hipdnn_golden_data_tests                        # both references
./bin/hipdnn_golden_data_tests --reference cpu        # host only, no GPU needed
./bin/hipdnn_golden_data_tests --gtest_filter='quick_*'
```

| Flag | Default | Purpose |
|---|---|---|
| `--reference cpu\|gpu\|both` | `both` | Which reference suites to register (`…_CpuRef`, `…_GpuRef`). |
| `--golden-data-dir`, `--gd` | as above | Bundle root. |
| `--validator` | `auto` (host) | Where the comparison runs. |

It has no skip path for a bundle it can run: a test is registered only when the
bundle has golden data **and** every node in its graph is in that reference's
required-op set, and the counts are printed at registration. An engine's TOML
never applies here. A tree that has not run `dvc pull` registers nothing and says
so. It is registered once, not per provider.

## Test tiers

Tiers bound how long a run takes. A bundle's tier is its top-level directory; a
C++ test's tier is its GTest instantiation prefix.

| Tier | Bundle directory → suite prefix | C++ GTest prefix | CTest label | Typical cadence |
|---|---|---|---|---|
| Quick | `quick/` → `quick_*` | `Smoke` (or none — see below) | `quick` | every PR |
| Standard | `standard/` → `standard_*` | `Standard` | `standard` | PR gate |
| Comprehensive | `comprehensive/` → `comprehensive_*` | `Comprehensive` | `comprehensive` | nightly |
| Full | `full/` → `full_*` | `Full` | `full` | weekly |

Timeouts come from `execution_settings.category_timeouts` in each YAML file and
differ per file.

### How tiers cascade

A higher tier includes the lower ones, but only because each YAML category
**lists** the lower tiers' patterns — CTest labels do not inherit. In the
provider files, `standard` lists `quick_*` *and* `standard_*`, so:

```
ctest -L quick           →  quick
ctest -L standard        →  quick + standard
ctest -L comprehensive   →  quick + standard + the comprehensive patterns
ctest -L full            →  everything ("*")
```

Check the file you are relying on; a category that forgets a lower tier's
pattern silently drops it.

### Smoke is a catch-all

For C++ tests, the quick tier is often written as an exclusion
(`-Standard*:Comprehensive*:Full*`), so any test *without* a tier prefix lands in
quick. If quick starts timing out, look for a large shape that forgot its
prefix. In GTest filter syntax only the first `-` starts the negative list;
`:-` between patterns does not negate.

## What decides whether a test runs

Six independent layers, outside-in. A test runs only if it survives all of them.

| # | Layer | Decided | Effect |
|---|---|---|---|
| 1 | Build options (`-DBUILD_CPP_GRAPH_TESTS`, op enables) | configure | A test not compiled cannot be selected. C++ graph tests are **off by default**. |
| 2 | `exclude_gpu` in the tier YAML | configure | Extra negative patterns per arch; a `"*"` pattern registers the suite `DISABLED` (`***Not Run (Disabled)`). |
| 3 | `ctest -L` / `-LE` / `-R` | ctest invocation | Selects suites by label or name. |
| 4 | `--gtest_filter` (from the YAML, or yours) | binary start | Selects cases by GTest name. |
| 5 | Metadata guards and TOML `test_skips` | each test's `SetUp()` | Skip on VRAM, arch-locked golden data, or a recorded engine limitation. |
| 6 | The engine's ranked-engine query | each test body | The engine declines the graph → the case is skipped (or fails if a claim promised support). |

## Verification modes

Once an engine runs a bundle, its output is compared against an oracle chosen by
`--verification-mode`:

| Mode | Oracle |
|---|---|
| `auto` (default) | golden data → GPU reference → CPU reference → skip, first available wins |
| `golden` | golden data only; **fails** a bundle with no golden data |
| `gpu` | the GPU reference executor (no DVC pull needed) |
| `cpu` | the CPU reference executor (no DVC pull needed) |

The comparison runs where the expected values live: device-side for a GPU
reference, host-side for a CPU reference or golden data. `--validator cpu|gpu`
overrides that for the whole run; only the pass/fail decision moves — failure
reports are always built on the host. The retired `golden-check` mode is now the
`hipdnn_golden_data_tests` binary.

A bundle's `enforcement_level` metadata can stop verification short on purpose:
`applicability` passes once the engine accepts the graph, `buildable` once its
plans compile.

## Reading the output

A run of `hipdnn_integration_tests` ends with up to four blocks on stderr. Read
all of them; the exit code alone is not a result.

### Test coverage summary

```text
==== TEST COVERAGE SUMMARY ====
Passed:  2457 / 5638 (43.6%)
Skipped: 3181
Failed:  0
```

**Report the three counts, not "exit 0".** A run that skips every case is green.
To tell an expected skip from a regression, read the skip reasons:

- `[arch <gcnArchName>] <reason>` — a TOML `test_skips` entry: a recorded,
  known limitation.
- `Engine could not execute bundle "<path>": Engine <NAME> does not support this
  graph (…)` — the engine declined the graph. With no claim on that cell this is
  only a skip; with a claim it is a `CLAIM_BROKEN` failure.
- `Bundle requires N MB VRAM …` / `Golden data generated on <arch> …` —
  metadata guards: this machine cannot host the case.

### Unverifiable bundles

```text
==== REFERENCE EXECUTOR ERRORS (n) ====
==== UNVERIFIABLE BUNDLES (n) ====
```

Bundles that ran on the engine but had no working oracle (no golden data and the
reference could not run the graph, or the reference errored). They skip, so they
do not change the exit code — but nothing checked their output.

### Support-claim summary

Printed by every run that is not a `--write-support-claims` run:

```text
==== SUPPORT CLAIM SUMMARY (ENFORCING) ====
{ "support_claim_summary": { … } }
```

The header says `ENFORCING` or `WARNING ONLY -- NOT ENFORCED`. The JSON is
stable (sorted, every key always present) so a tool can parse it. The keys that
matter to a developer:

| Key | Meaning | What to do |
|---|---|---|
| `claim_failures` | `CLAIM_BROKEN`: a sidecar claims the engine accepts this graph on this arch/platform, and it no longer does. `QUERY_ERRORED`: the query failed, so acceptance is unknown. These fail the run. | Fix the engine, or — if dropping support is intended — retract the claim in a reviewed change. |
| `failed_in_use` | The engine accepted the graph, then the test failed (execution or comparison). Not a claim failure; the run is already red for the real reason. | Debug the failure. Do **not** add this cell to a sidecar. |
| `unclaimed_support` | The engine accepts graphs that no sidecar claims — support that exists but is not written down. `reached` / `required` say how far each got. | **Update the sidecars** — see [Updating support claims](adding-tests.md#updating-support-claims). |
| `unenforced.no_applicable_claim` | Sidecars were read but promise nothing for this arch/platform. | If it is every claim-bearing graph (typical on a bring-up ASIC), this run enforced nothing; add claims for this arch. |
| `unenforced.not_selected`, `skipped_before_run`, `not_opened` | Claim-bearing graphs this run did not check: filtered out, skipped in `SetUp()`, or failed to open. | Expected on a filtered lane; otherwise investigate. |
| `harness_defects` | Anything above zero is a harness bug. | Report it. |
| `verdicts.confirmed` | Claims that held **and** reached the depth their bundle's `enforcement_level` requires. `accepted` is weaker: the engine took the graph but nothing verified the result. | A published support matrix should carry `confirmed`. |

Full key list, the coverage ladder, and the harness lifecycle behind the
verdicts: [Support Claim Enforcement](support-claim-enforcement.md#reading-the-summary).

Claims are checked only for the engine under test, only on the running arch and
platform, and only for bundles — C++ graph tests carry no sidecars.

### Hard stops

| Message | Meaning |
|---|---|
| `Error: zero tests ran.` followed by `registered: N … selected: 0 …` | Nothing matched. `registered: 0` is a discovery problem (wrong plugin, wrong `--golden-data-dir`); `N` registered with 0 selected is a filter problem. Always a configuration bug. Exit 1. |
| `FATAL: --enforce-support-claims is active and N graph(s) carrying support claims actually ran, but not one of them was ever queried.` | Enforcement verified nothing. Usually every claim-bearing graph failed to open. Exit 1. |
| `support claims exist for X but were never queried` | A code path skipped the claim query: a harness bug, not a data problem. |
| `Error: --enforce-support-claims requires --test-engine …` | No engine named to check claims against. |

The zero-tests guard fires only when `--test-engine` was given or the bundle
root exists; a bare local run with neither may legitimately run empty.

## Runs that look green but are not

| Symptom | Why it is not a pass |
|---|---|
| `ctest -L <label>` prints `No tests were found!!!`, exit 0 | CTest's default `--no-tests=ignore`. A typo'd label, the wrong build directory, or a tier whose only suites are `DISABLED`. Add `--no-tests=error` when scripting. |
| `[  PASSED  ] 0 tests.` from GTest | A filter matched nothing. `hipdnn_integration_tests` turns this into `Error: zero tests ran.`; other binaries do not. |
| `Skipped` equals the total | The engine declined everything, or a TOML skip covers everything. Read the skip reasons. A claim-free graph the engine silently stopped supporting looks exactly like this. |
| `SUPPORT CLAIM SUMMARY` with `no_applicable_claim` equal to every claim-bearing graph | Claims exist, but none for this arch/platform — nothing was enforced. |
| Header says `WARNING ONLY -- NOT ENFORCED` | No `--test-engine`, or `--enforce-support-claims=false`: broken claims were reported, not failed. |
| A C++ graph test "does not run" | Expected: C++ graph tests build only with `-DBUILD_CPP_GRAPH_TESTS=ON`. |
| `auto` mode, no `dvc pull` | Golden comparison silently fell back to a reference executor. Force `--verification-mode golden` to require golden data. |
| `hipdnn_golden_data_tests` registers nothing | Golden `.bin` blobs were not pulled. |

## In CI

- `.github/workflows/hipdnn-superbuild-ci.yml` configures with
  `-DROCM_LIBS_ENABLE_ROOT_CTEST=ON` and runs `ctest --test-dir build
  --output-on-failure` — every registered lane, enforcing claims. It does not
  `dvc pull`, so golden comparisons there fall back to references and
  `hipdnn_golden_data_tests` registers nothing.
- `.github/workflows/therock-ci-linux.yml` runs a bare `dvc pull` before
  building, so golden data is present in those lanes.
- The `verify-support-claims` pre-commit hook validates every sidecar on each
  commit that touches `integration-test-bundles/`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Engine 'X' is not loaded. Check the plugin path.` | Wrong `--test-article`, or `--test-engine` misspelled. |
| `Error: Article path does not exist` / `Config path does not exist` | Fix the path; both are resolved before anything loads. |
| Bundle tests do not register | `--no-bundles` / `HIPDNN_TEST_ALLOW_BUNDLES=0`, or the bundle root is missing (see the `bundle data dir:` line). |
| `Bundle name collision: '…' produced by both:` | Two bundle paths sanitize to the same GTest name; rename one. |
| `verification-mode=golden was requested but this bundle has no golden data` | `dvc pull` the op, or use `--verification-mode=auto`. |
| `verification-mode 'golden-check' has been retired` | Run `hipdnn_golden_data_tests`, and unset `HIPDNN_TEST_VERIFICATION_MODE`. |
| `STATUS_DLL_NOT_FOUND` / `0xc0000135` on Windows | Put the ROCm and build `bin` directories on `PATH`. |
