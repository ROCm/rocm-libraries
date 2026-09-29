# Running the Integration Tests

How to run the cross-provider suite against an engine, how the tiers and
filters decide what runs, and how to read what it prints — including the ways a
run can look green while testing nothing. Terms such as *lane*, *oracle* and
*cell* are defined in the [index](README.md#terms).

- File formats named here are described in [File Formats](file-formats.md).
- Acting on what a run reports (new bundles, stale claims) is in
  [Adding Tests and Updating Claims](adding-tests.md).

Commands below run from the repository root of a superbuild whose build
directory is `build/`. There the suite's binaries are in `build/bin/`, and the
bundle tree the tests read is a copy in `build/lib/integration-test-bundles/`
(see [The build reads a copy of the bundles](#the-build-reads-a-copy-of-the-bundles)).

## The binaries

| Binary | Tests | Engine loaded |
|---|---|---|
| `hipdnn_integration_tests` | The bundles (plus any C++ tests compiled in) against **one provider's engine** | yes |
| `hipdnn_golden_data_tests` | Our checked-in golden `.bin` data against the CPU/GPU reference executors | **no** |
| `hipdnn_gpu_ref_tests` | The GPU/CPU reference executors themselves | no |
| `hipdnn_integration_tests_unit_tests` | The harness (discovery, TOML, claims, verdicts) — no device needed | no |

`hipdnn_integration_tests` is built once in `dnn-providers/integration-tests/`
and run by **each provider** against its own plugin. That is how one bundle
tree validates every engine.

## Three ways to run it

### 1. CTest by tier (what CI does)

Each lane registers one CTest suite per tier from its provider's tier YAML
([format](file-formats.md#tier-yaml-files--test_categoriesyaml)). The suites
live in the **provider's** build directory:

```bash
ctest --test-dir build/dnn-providers/miopen-provider -L quick --output-on-failure
ctest --test-dir build/dnn-providers/miopen-provider -L quick -N          # list, do not run
ctest --test-dir build/dnn-providers/miopen-provider -R external-integration_quick -N -V  # print the exact command lines
```

From the superbuild root (`ctest --test-dir build`) this only works when the
build was configured with `-DROCM_LIBS_ENABLE_ROOT_CTEST=ON` (CI sets it; the
default is off). Without it, CTest finds nothing — and exits 0.

Every lane's suites carry the labels `integration_test`,
`external_integration_test`, `slow` and the engine name, plus the tier labels
from the YAML. Two `-L` flags must both match:

```bash
ctest --test-dir build/dnn-providers/miopen-provider -L quick -L MIOPEN_ENGINE
```

Suite names are `<prefix>_<category>_suite`, where the provider sets the prefix —
for example `miopen-provider-external-integration_quick_suite`. A YAML with
`exclude_gpu` entries adds per-arch variants named
`<prefix>_<category>_<arch>_suite`; see [layer 2](#what-decides-whether-a-test-runs).

#### Seeing the output of a passing run

`--output-on-failure` prints a test's output only when that test fails. A green
lane therefore shows neither the coverage summary nor the support-claim summary
— and `unclaimed_support` never fails a test. To read them:

- run with `-V` (`ctest ... -L quick -V`), or
- read `build/dnn-providers/<provider>/Testing/Temporary/LastTest.log` after the run, or
- use the check target or the binary directly (below), which always print them.

### 2. The provider's check target (no tier filter)

```bash
cmake --build build --target miopen-provider-external-integration-check
```

Runs the lane without a tier filter and always prints its output. It is the way
to see an engine's full picture, and the wrong way to reproduce a tier.

| Provider | Target | Engine | Note |
|---|---|---|---|
| miopen-provider | `miopen-provider-external-integration-check` | `MIOPEN_ENGINE` | |
| hipblaslt-provider | `hipblaslt-provider-external-integration-check` | `HIPBLASLT_ENGINE` | runs with `--gtest_filter=*Matmul*` |
| hip-kernel-provider | `hip-kernel-provider-external-integration-check` | `HIP_MLOPS_ENGINE` | |
| hip-kernel-provider | `hip-kernel-provider-asm-sdpa-external-integration-check` | `ASM_SDPA_ENGINE` | |
| hip-kernel-provider | `hip_kernel_provider_asm_sdpa_gpu_ref_integration_tests` | `ASM_SDPA_ENGINE` | GPU reference, C++ SDPA tests only, no tiers |

A lane registered without a tier YAML — like the last row — is a single CTest
test with no tier labels. The authoritative list is the set of
`add_external_integration_test_target()` calls under `dnn-providers/`; adding
one is described in [Add an engine lane](adding-tests.md#add-an-engine-lane).

### 3. The binary directly (debugging one case)

Pass the plugin, pin the engine, and pass that engine's TOML — what the CTest
lanes pass. The quickest way to get the exact plugin path and arguments is
`ctest -N -V` on the lane's suite (above).

```bash
build/bin/hipdnn_integration_tests \
    --test-article <path to the provider plugin, e.g. libmiopen_plugin.so> \
    --test-engine  MIOPEN_ENGINE \
    --test-config  dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml \
    --gtest_filter='quick_RMSNorm_Default.*'
```

- Always pass `--test-engine` when testing one provider. Without it, hipDNN's
  normal engine selection picks the winner, a "pass" may have come from a
  different engine, and no support-claim summary is produced.
- Add `--golden-data-dir dnn-providers/integration-tests/integration-test-bundles`
  to read the source tree instead of the build copy — useful right after
  editing bundles or sidecars.
- On Windows the binary needs the ROCm `bin` directory and `build/bin` on
  `PATH`; without them it dies with `STATUS_DLL_NOT_FOUND` (`0xc0000135`)
  before running anything.
- Standard GTest sharding (`GTEST_TOTAL_SHARDS` / `GTEST_SHARD_INDEX`) works.

### The build reads a copy of the bundles

CMake copies `integration-test-bundles/` into `build/lib/integration-test-bundles/`
**at configure time only**, and that copy is what the lanes, check targets and
`hipdnn_golden_data_tests` read. After you add or edit a bundle or sidecar, or
run `dvc pull`, re-run CMake (or pass `--golden-data-dir` pointing at the source
tree) before running — otherwise you are testing the old tree.

## Command-line reference — `hipdnn_integration_tests`

| Flag | Env var | Default | Purpose |
|---|---|---|---|
| `--test-article`, `--ta` | | plugin discovery | Path to the engine plugin (`.so` / `.dll`). Only that plugin is loaded. |
| `--test-engine`, `--te` | | none | Pin the run to one engine. An engine that is not loaded exits 1 before any test: `Error: Engine '<name>' is not loaded. Check the plugin path.` |
| `--test-config`, `--tc` | | none | The engine's TOML ([format](file-formats.md#per-engine-test-config--configengine_nametoml)). A missing path or invalid TOML exits 1. |
| `--verification-mode`, `--vm` | `HIPDNN_TEST_VERIFICATION_MODE` | `auto` | How bundle output is checked; see [Verification modes](#verification-modes). |
| `--validator` | `HIPDNN_TEST_VALIDATOR` | `auto` | Where the comparison runs: `auto` (follow the reference), `cpu`, `gpu`. |
| `--reference-executor` | `HIPDNN_TEST_REFERENCE_EXECUTOR` | `cpu` | Reference for C++ graph tests only. |
| `--golden-data-dir`, `--gd` | `HIPDNN_TEST_GOLDEN_DATA_DIR` | `<exe>/../lib/integration-test-bundles/` | Bundle root. The flag's path must exist; the env var's is not checked up front. |
| `--no-bundles` | `HIPDNN_TEST_ALLOW_BUNDLES` | bundles on | Register only the C++ tests compiled into the binary. The env var, when set, wins over the flag in either direction. |
| `--enforce-support-claims[=true\|false]` | | on | Fail a test whose claim the engine breaks. See [Support-claim summary](#support-claim-summary). |
| `--write-support-claims` | | off | Authoring run: record which graphs the engine accepts into sidecars. See [Adding Tests](adding-tests.md#record-claims-with---write-support-claims). |
| `--skip-graph-validation` | | off | Pass as soon as the engine accepts the graph. **C++ graph tests only**; bundles ignore it (their equivalent is `enforcement_level`). |
| `--fail-on-unsupported` | | off | Fail instead of skip when no engine supports a graph. **C++ graph tests only.** |
| `--generate-support-matrix[=file]` | | off | Write a markdown matrix. **C++ graph tests only**; on a default build it writes an empty table. |
| `--capture-bundles <dir>` | | off | Dump compiled-in C++ graph tests as bundle JSON (see [Adding Tests](adding-tests.md#from-an-existing-c-graph-test)). |
| `--gtest_*` | | | Passed through to GTest. |

Flag combinations the binary refuses (exit 1):

- `--enforce-support-claims` or `--enforce-support-claims=true` without
  `--test-engine`.
- `--write-support-claims` together with `--enforce-support-claims` or `=true`.
- `--write-support-claims` without `--test-article`, or without a bundle root
  (`--golden-data-dir` or `HIPDNN_TEST_GOLDEN_DATA_DIR`).

`--enforce-support-claims=false` is accepted in every combination. When
enforcement is merely inherited and there is no `--test-engine`, the run
enforces nothing and prints no claim summary.

A machine with no HIP device prints `No HIP devices available; skipping …` and
exits 0 immediately, before reading any other flag.

## `hipdnn_golden_data_tests`

Recomputes every bundle that has golden data with a reference executor and
compares it against the checked-in `.bin` files. No plugin, no engine, no claims:
it validates **our data**, not a provider.

```bash
build/bin/hipdnn_golden_data_tests                        # both references
build/bin/hipdnn_golden_data_tests --reference cpu        # host only, no GPU needed
build/bin/hipdnn_golden_data_tests --gtest_filter='quick_*'
```

| Flag | Default | Purpose |
|---|---|---|
| `--reference cpu\|gpu\|both` | `both` | Which reference suites to register (`…_CpuRef`, `…_GpuRef`). |
| `--golden-data-dir`, `--gd` | as above | Bundle root. |
| `--validator` | `auto` (host) | Where the comparison runs. `gpu` needs a device, even for the CPU suite. |
| `--test-config`, `--tc` | none | Accepted, but this harness applies neither the TOML's skips nor its tolerances. |

It has no skip path for a bundle it can run: a test is registered only when the
bundle has golden data **and** every node in its graph is in that reference's
required-op set (`src/harness/bundle/ReferenceOpCoverage.hpp`); the counts are
printed at registration. A tree whose `.bin` files were never pulled registers
nothing and says so. It is registered once, not per provider.

## Test tiers

Tiers bound how long a run takes. A bundle's tier is its top-level directory
(`quick/`, `standard/`, `comprehensive/`, `full/`), which becomes the prefix of
its GTest suite name. A C++ test's tier is its GTest instantiation prefix.

What each CTest tier selects in the provider lanes (miopen, hipblaslt and
HIP_MLOPS files; the ASM_SDPA lane uses `*Sdpa*` for both comprehensive and
full):

| CTest label | Bundle suites | C++ tests | Other labels on the suite |
|---|---|---|---|
| `quick` | `quick_*` | `Smoke/*`, `Quick/*` | `pre-commit`, `smoke` |
| `standard` | `quick_*`, `standard_*` | `Smoke/*`, `Quick/*`, `Standard/*` | `pr`, `precheckin` |
| `comprehensive` | `quick_*`, `standard_*`, `comprehensive_*`, `full_*` (HIP_MLOPS excludes `full_Layernorm_*`) | the above, `Comprehensive/*`, and untiered `Integration*` | `nightly`, `extended` |
| `full` | everything (`*`) | everything | `all` |

Timeouts come from `execution_settings.category_timeouts` in each YAML file and
differ per file.

### How tiers cascade

A higher tier includes the lower ones only because each YAML category **lists**
the lower tiers' patterns — CTest labels do not inherit:

```
ctest -L quick           →  quick
ctest -L standard        →  quick + standard
ctest -L comprehensive   →  quick + standard + comprehensive + full bundles, plus untiered C++ tests
ctest -L full            →  everything
```

A category that forgets a lower tier's pattern silently drops it; check the
file you rely on. Two consequences for C++ tests in the provider lanes: a test
must be instantiated under `Smoke`, `Quick` or `Standard` to run in PR tiers,
and a test with **no** tier prefix runs only from `comprehensive` up.

### The project's own binaries use a catch-all

`dnn-providers/integration-tests/test_categories.yaml` (for
`hipdnn_gpu_ref_tests`, the harness unit tests and `hipdnn_golden_data_tests`)
writes its quick tier as an exclusion — `*` minus `Standard*`,
`Comprehensive*` and `Full*` — so there any test without a tier prefix lands in
quick. If that quick tier starts timing out, look for a large shape that forgot
its prefix. In GTest filter syntax only the first `-` starts the negative list;
`:-` between patterns does not negate.

## What decides whether a test runs

Six independent layers, outside-in. A test runs only if it survives all of them.

| # | Layer | Decided | Effect |
|---|---|---|---|
| 1 | Build options (`-DBUILD_CPP_GRAPH_TESTS`, op enables) | configure | A test not compiled cannot be selected. C++ graph tests are **off by default**. |
| 2 | `exclude_gpu` in the tier YAML | configure (suites) and `ctest` (selection) | Adds per-arch variant suites (`…_<category>_<arch>_suite`, label `ex_gpu_<arch>`) carrying that arch's negative patterns, or `DISABLED` for `"*"`. You pick the variant with `-L ex_gpu_<arch>`; a plain `-L quick` still runs the unfiltered suite too. See `shared/ctest/README.md`. |
| 3 | `ctest -L` / `-LE` / `-R` | ctest invocation | Selects suites by label or name. |
| 4 | `--gtest_filter` (from the YAML, or yours) | binary start | Selects cases by GTest name. |
| 5 | Metadata guards and TOML `test_skips` | each test's `SetUp()` | Skip on VRAM, arch-locked golden data, or a recorded engine limitation. |
| 6 | The engine's ranked list | each test body | The engine declines the graph → the case is skipped (or fails if a claim promised support). |

A bundle that fails to load (bad JSON, a sweep case with a missing placeholder
or metadata block, a bad `golden.path`) never reaches these layers: it is
dropped from registration with an `ERROR` log line, and the test count is
simply lower. See [File Formats](file-formats.md#where-each-format-is-enforced).

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
overrides that for the whole run; `gpu` then needs a device even for golden or
CPU-reference comparisons. Only the pass/fail decision moves — failure reports
are always built on the host. The retired `golden-check` mode is now the
`hipdnn_golden_data_tests` binary.

A bundle's `enforcement_level` metadata can stop verification short on purpose:
`applicability` passes once the engine accepts the graph, `buildable` once its
plans compile.

## Reading the output

A run of `hipdnn_integration_tests` ends with up to four blocks, in this order.
The first two go to **stdout**, the last two to **stderr**; capture both.

### Unverifiable bundles

```text
==== REFERENCE EXECUTOR ERRORS (n) ====
==== UNVERIFIABLE BUNDLES (n) ====
```

- **Unverifiable bundles** ran on the engine with no oracle available (no golden
  data, and no reference executor supports the op). They skip. Nothing checked
  their output.
- **Reference executor errors** are bugs in a reference executor that should
  have handled the op. In `auto` mode a GPU-reference error falls back to the
  CPU reference; an error in the last oracle tried — or in an explicit
  `--verification-mode gpu|cpu` — fails the test.

### Support-claim summary

```text
==== SUPPORT CLAIM SUMMARY (ENFORCING) ====
{ "support_claim_summary": { … } }
```

Printed when the run pinned an engine with `--test-engine` and found at least
one bundle with a sidecar (or recorded a verdict); not printed by
`--write-support-claims` runs. The header says `ENFORCING` or
`WARNING ONLY -- NOT ENFORCED`. The JSON is sorted, so two runs over the same
tree print the same document. The keys that matter to a developer:

| Key | Meaning | What to do |
|---|---|---|
| `claim_failures` | `CLAIM_BROKEN`: a sidecar claims the engine accepts this graph on this arch and platform, and it no longer does. `QUERY_ERRORED`: the support query failed, so acceptance is unknown. These fail the test when enforcing. | `CLAIM_BROKEN`: fix the engine or, if dropping support is intended, [retract the claim](adding-tests.md#retract-a-claim). `QUERY_ERRORED`: investigate the error; never retract over it. |
| `failed_in_use` | A **claimed** cell: the engine accepted the graph, then the test failed (execution or comparison). The run is already red for the real reason. | Fix the engine, or retract the claim until it is fixed. |
| `unclaimed_support` | The engine accepts graphs that no sidecar claims — support that exists but is not written down. `reached` / `required` say how far each got. Only bundles that have a sidecar can appear here. Does not fail the run. | **Update the sidecars** — see [Updating support claims](adding-tests.md#updating-support-claims). Be wary of entries whose `reached` is below `required`: the engine accepts them but nothing verified the result. |
| `unenforced.no_applicable_claim` | Graphs whose sidecar was read but claims nothing for this arch and platform (empty sidecars count here). | If `verdicts` are all zero, this run enforced nothing — expected on a new arch or where sidecars are still empty. |
| `unenforced.not_selected`, `skipped_before_run`, `not_opened` | Graphs with sidecars that this run did not check: filtered out, skipped in `SetUp()`, or failed to open. | Expected on a filtered lane; otherwise investigate. |
| `harness_defects` | Anything above zero is a harness bug. | File an issue against the integration-test harness with the summary attached. |
| `verdicts.confirmed` / `verdicts.accepted` | Counts. `confirmed`: claims that held **and** reached the depth their bundle's `enforcement_level` requires. `accepted`: the engine took the graph but nothing verified the result (look in `UNVERIFIABLE BUNDLES`). | A published support matrix should carry `confirmed`. |

`not_selected`, `skipped_before_run` and `harness_defects.missed_query` appear
only when `counters_consistent` is `true`. Full key list, the coverage ladder,
and the harness lifecycle behind the verdicts:
[Support Claim Enforcement](support-claim-enforcement.md#reading-the-summary).

Claims are checked only for the engine under test, only on the running arch and
platform, and only for bundles — C++ graph tests carry no sidecars.

### Test coverage summary

```text
==== TEST COVERAGE SUMMARY ====
Passed:  2457 / 5638 (43.6%)
Skipped: 3181
Failed:  0
```

**Check all three counts, not just the exit code.** A run that skips every case
is green. To tell an expected skip from a regression, read the skip reasons:

- `[arch <gcnArchName>] <reason>` — a TOML `test_skips` entry: a recorded,
  known limitation.
- `Engine could not execute bundle "<path>": Engine <NAME> does not support this
  graph (…)` — the engine declined the graph. With no claim on that cell this is
  only a skip; with a claim it is a `CLAIM_BROKEN` failure.
- `Bundle requires N MB VRAM …` / `Golden data generated on <arch> …` —
  metadata guards: this machine cannot host the case.

### Hard stops

| Message | Meaning |
|---|---|
| `Error: zero tests ran.` followed by `registered: N … selected: 0 …` | Nothing matched. `registered: 0` is a discovery problem (wrong plugin, wrong bundle root); `N` registered with 0 selected is a filter problem. Always a configuration bug. Exit 1. |
| `FATAL: --enforce-support-claims is active and N graph(s) carrying support` (continues over several lines) | Graphs with sidecars ran but none was ever queried: enforcement verified nothing. Usually every one of them failed to open. Exit 1. |
| `support claims exist for X but were never queried` | A code path skipped the claim query: a harness bug, not a data problem. |
| `Error: --enforce-support-claims requires --test-engine …` | No engine named to check claims against. |

The zero-tests guard fires only when bundles are enabled **and** either
`--test-engine` was given or the bundle root exists.

## Runs that look green but are not

| Symptom | Why it is not a pass |
|---|---|
| `No HIP devices available; skipping …`, exit 0 | No GPU: nothing ran, no claim was checked. |
| `ctest -L <label>` prints `No tests were found!!!`, exit 0 | CTest's default `--no-tests=ignore`. A typo'd label, the wrong build directory, or a tier whose only suites are `DISABLED`. Add `--no-tests=error` when scripting. |
| A green `ctest --output-on-failure` run | The summaries were not printed at all; see [Seeing the output of a passing run](#seeing-the-output-of-a-passing-run). |
| `WARNING: Bundle tests are enabled but …` / `WARNING: No bundles could be loaded from …` | No bundles registered; only compiled-in C++ tests ran. |
| Fewer tests than expected, with `ERROR` lines at registration | Bundles that failed to load were dropped, not failed. |
| `[  PASSED  ] 0 tests.` from GTest | A filter matched nothing. `hipdnn_integration_tests` turns this into `Error: zero tests ran.` when bundles are enabled; with `--no-bundles`, and in other binaries, it stays green. |
| `Skipped` equals the total | The engine declined everything, or a TOML skip covers everything. Read the skip reasons. A claim-free graph the engine silently stopped supporting looks exactly like this. |
| `verdicts` all zero while `no_applicable_claim` > 0 | Sidecars exist but claim nothing for this arch/platform — nothing was enforced. |
| Header says `WARNING ONLY -- NOT ENFORCED` | `--enforce-support-claims=false`: broken claims were reported, not failed. |
| A run without `--test-engine` | No claim summary at all, and results may come from a different engine. |
| A C++ graph test "does not run" | Expected: C++ graph tests build only with `-DBUILD_CPP_GRAPH_TESTS=ON`. |
| New bundles or sidecars do not show up | The build reads a configure-time copy; re-run CMake. |
| `auto` mode, no `dvc pull` | Golden comparison silently fell back to a reference executor. Force `--verification-mode golden` to require golden data. |
| `hipdnn_golden_data_tests` registers nothing | Golden `.bin` blobs were not pulled (or were pulled after configure). |

## In CI

- `.github/workflows/hipdnn-superbuild-ci.yml` configures with
  `-DROCM_LIBS_ENABLE_ROOT_CTEST=ON` and runs `ctest --test-dir build
  --output-on-failure` — every registered lane, enforcing claims. It does not
  `dvc pull`, so golden comparisons there fall back to references and
  `hipdnn_golden_data_tests` registers nothing.
- `.github/workflows/therock-ci-linux.yml` runs a bare `dvc pull` before
  building, so golden data is present in those lanes.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Engine 'X' is not loaded. Check the plugin path.` | Wrong `--test-article`, or `--test-engine` misspelled. |
| `Error: Article path does not exist` / `Config path does not exist` | Fix the path; both are resolved before anything loads. |
| `WARNING: Bundle tests are enabled but the data directory does not exist: <dir>` | Wrong `--golden-data-dir` / `HIPDNN_TEST_GOLDEN_DATA_DIR`, or the build has no bundle copy (re-run CMake). |
| `WARNING: Bundle tests are enabled but no bundles were found in <dir>` / `WARNING: No bundles could be loaded from <dir>` | The root is empty, or every bundle failed to load — look for `ERROR` lines above. |
| Bundles missing only in this run | `--no-bundles` or `HIPDNN_TEST_ALLOW_BUNDLES=0`. |
| `Bundle name collision: '…' produced by both:` | Two bundle paths sanitize to the same GTest name; rename one. |
| `verification-mode=golden was requested but this bundle has no golden data` | `dvc pull` the op and re-run CMake, or use `--verification-mode=auto`. |
| `verification-mode 'golden-check' has been retired` | Run `hipdnn_golden_data_tests`, and unset `HIPDNN_TEST_VERIFICATION_MODE`. |
| `STATUS_DLL_NOT_FOUND` / `0xc0000135` on Windows | Put the ROCm `bin` and `build/bin` directories on `PATH`. |
