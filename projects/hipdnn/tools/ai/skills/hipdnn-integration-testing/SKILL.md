---
name: hipdnn-integration-testing
description: Domain reference for hipDNN's cross-provider integration test suite (dnn-providers/integration-tests) — the bundle/sweep vs C++ test split, YAML tier filtering, per-engine TOML config, CMake/CTest wiring, direct hipdnn_integration_tests invocation against an engine, and the failure modes that let a broken suite report green. Use when adding, running, configuring, or triaging integration tests, or when a run looks suspicious (0 tests, all-skip, "did engine X regress" questions).
argument-hint: "[engine: MIOPEN_ENGINE|HIPBLASLT_ENGINE|ASM_SDPA_ENGINE|HIP_MLOPS_ENGINE|...] [topic: run|toml-config|yaml-tiers|cmake-wiring|zero-tests|support-claims|add-test]"
allowed-tools: Bash, Read, Grep, Glob
---

# hipDNN Integration Testing Reference

This is a knowledge skill, not an automation script. It explains how
`dnn-providers/integration-tests` is put together so you can configure it,
run its binary directly against a specific engine, and correctly read the
output — including the cases where the suite reports success while coverage
is silently missing. To actually invoke ctest/ninja targets inside a
superbuild, pair this with the `hipdnn-superbuild-test` skill; that skill
does target discovery and process invocation, this one supplies the domain
knowledge behind what those targets mean.

Ground truth for everything below lives in
`dnn-providers/integration-tests/README.md` and
`projects/hipdnn/docs/rfcs/0006_PluginAgnosticIntegrationTests.md` /
`0011_GoldenReferenceValidation.md` / `0015_EngineSupportClaims.md`. If code
and this skill disagree, re-read the source before trusting either.

## 1. Two mechanisms — pick one, don't guess

| | Bundles + sweeps (default, CI driver) | C++ integration tests (special cases only) |
|---|---|---|
| What it is | Graph JSON + a `sweep.json` case matrix | `buildGraph()` + `INSTANTIATE_TEST_SUITE_P` |
| Add a case | Edit JSON / `import_graph.py` — no compile | Write C++, recompile |
| Built by default | Yes (registration on by default) | **No** — gated behind `-DBUILD_CPP_GRAPH_TESTS=ON` |
| Use for | "does this graph run and match a reference on this engine" | error/unhappy paths, API-contract behavior, serialization round-trips, benchmarking knobs, determinism, pass-by-value semantics — anything that is *not* "run a graph and verify output" |

**New graph-verification coverage must be a bundle.** CMake enforces this:
every `.cpp` under `src/integration-tests/` must be registered through
exactly one of `add_cpp_graph_test_sources()` (builds only with
`BUILD_CPP_GRAPH_TESTS=ON`) or `add_always_built_test_sources()` (always
builds, and additionally requires listing the file in
`HIPDNN_IT_ALWAYS_BUILT_SOURCES` at the top of
`src/integration-tests/CMakeLists.txt`). A file registered through neither is
a configure-time `FATAL_ERROR` naming the file and pointing at bundles — this
is by design, not a bug to work around.

If a proposed C++ test is really "build graph X, run it, compare to a
reference," it belongs in a bundle. Convert with
`migration-scripts/import_graph.py` / `--capture-bundles` rather than adding
another `INSTANTIATE_TEST_SUITE_P`.

## 2. Directory map

```
dnn-providers/integration-tests/          # shared, plugin-agnostic
  CMakeLists.txt                          # builds hipdnn_integration_tests
  test_categories.yaml                    # tiers for this project's OWN binaries
  test_categories_external.yaml           # tiers for pre-registered meta-tests
  cmake/HipdnnIntegrationTestHelpers.cmake# add_external_integration_test_target()
  src/main.cpp                            # CLI parsing, shared handle, RUN_ALL_TESTS
  src/harness/                            # TestConfig, bundle discovery, support claims
  src/integration-tests/{op}/              # shared cross-provider C++ tests (opt-in build)
  tests/                                  # hipdnn_integration_tests_unit_tests, gpu-ref/
  integration-test-bundles/{tier}/{op}/   # bundle + sweep JSON, DVC-tracked tensors

dnn-providers/<provider>/                 # miopen-provider, hipblaslt-provider, hip-kernel-provider
  CMakeLists.txt                          # find_package(hipdnn_integration_tests); add_external_integration_test_target(...)
  config/<ENGINE_NAME>.toml               # this provider's --test-config file
  test_categories_integration.yaml        # tiers applied ONLY to the external hipdnn_integration_tests run
  integration_tests/                      # provider-local C++ tests (native, not the shared binary)
```

Three distinct `test_categories*.yaml` scopes exist — do not confuse them:
1. `dnn-providers/integration-tests/test_categories.yaml` — the shared
   project's own binaries (`hipdnn_integration_tests_unit_tests`,
   `hipdnn_gpu_ref_tests`).
2. `dnn-providers/integration-tests/test_categories_external.yaml` — labels
   pre-registered (non-GTest) CTest tests such as the test-name-validation
   meta-test and the Python bundle verifier, so they still get picked up by
   `ctest -L <tier>`.
3. `dnn-providers/<provider>/test_categories_integration.yaml` — applied only
   when `hipdnn_integration_tests` is run **against that provider's plugin**
   (i.e. it filters the shared binary's bundle/sweep suite names, not the
   provider's own native `*_plugin_tests`/`*_plugin_integration_tests`).

## 3. YAML tier filtering

Each `test_categories:` entry defines `description`, `test_patterns`
(GTest-glob, matched literally including `/` and `_`), `exclude`, and
cumulative `labels`. `execution_settings.category_timeouts` sets per-tier
CTest timeouts.

Tiers cascade: `ctest -L quick` → smoke only; `ctest -L standard` → quick +
standard; `comprehensive` and `full` keep adding. Bundle dir tier prefixes
(`quick/`, `standard/`, `comprehensive/`, `full/`) and GTest prefixes (no
prefix/`Smoke`, `Standard`, `Comprehensive`, `Full`) mirror each other.

**Smoke is a catch-all.** The quick/smoke suite is built from an *exclusion*
filter (`-Standard*:Comprehensive*:Full*`), so anything that does not start
with `Standard`/`Comprehensive`/`Full` lands in smoke automatically —
including a newly-added large shape that forgot its tier prefix. If smoke
starts timing out, look for a missing tier prefix before assuming
infrastructure regressed.

GTest filter syntax gotcha: only a single leading `-` starts the negative
section (`-Standard*:Comprehensive*:Full*`); `:-` between patterns does not
negate the next one — the dash becomes a literal character.

`ffm-quick`/`ffm-full` labels exist for the fast-feedback mechanism and are
usually a curated, hand-picked subset (see the miopen-provider
`test_categories_integration.yaml` example: exact case-id strings, not
globs) — regenerating a sweep can silently rename those ids out from under
the FFM tier. If you touch a sweep those ids reference, re-verify with
`ctest -L ffm-quick -N`.

## 4. Per-engine TOML config (`--test-config`)

Each provider owns one `config/<ENGINE_NAME>.toml` (e.g.
`miopen-provider/config/MIOPEN_ENGINE.toml`,
`hipblaslt-provider/config/HIPBLASLT_ENGINE.toml`,
`hip-kernel-provider/config/{ASM_SDPA,HIP_MLOPS}_ENGINE.toml`) passed via
`--test-config`/`TEST_CONFIG`. It lets you override tolerances or skip tests
for that engine without recompiling:

```toml
[meta]
version = 1                 # required; unsupported/missing version => parse error, not silent ignore

[[tolerance_overrides]]
filters = ["Smoke/IntegrationGpuConvWrw3dBfp16.Correctness/14"]
atol = 1.19
rtol = 0.2

[[test_skips]]
archs   = ["gfx90a", "gfx10", "gfx11", "gfx12"]   # optional; substring match vs raw gcnArchName; omit = all archs
platforms = ["windows"]                            # optional; "windows"/"linux"; omit = all platforms
filters = ["*ConvFwdBiasActiv*"]
reason  = "ROCm/rocm-libraries#6979 — no engine has an applicable solution for ConvBiasActiv fusion"
```

- `filters` are GTest-style globs matched against the **full GTest name** —
  the same string a `--gtest_filter` would match.
- **`tolerance_overrides`: later entries win** when multiple filters match
  the same test — this is a "last write wins" merge.
- **`test_skips`: the first matching entry wins** — this is the opposite
  order from `tolerance_overrides`. Don't assume both lists resolve
  conflicts the same way when adding a new entry near an existing one; check
  which list you're editing.
- Applies to both bundle/sweep tests and C++ graph tests — the lookup is in
  the shared harness (`TestConfig`/`TestSettings`), not per test type.
- Full schema: `src/harness/TestSettings.hpp`.

## 5. CMake / CTest wiring

The shared project builds one binary, `hipdnn_integration_tests`, and
installs it as a CMake package exporting
`cmake/HipdnnIntegrationTestHelpers.cmake`. It is **not** registered as a
tiered ctest target inside `integration-tests/CMakeLists.txt` itself — the
bundle suites it carries exercise a specific plugin, so each provider wires
its own run via `add_external_integration_test_target()`:

```cmake
if(NOT TARGET hipdnn_integration_tests)
    find_package(hipdnn_integration_tests CONFIG QUIET)   # standalone provider build
endif()

if(TARGET hipdnn_integration_tests)                       # superbuild target already present, or found above
    add_external_integration_test_target(
        TARGET_NAME    ${PROJECT_NAME}-external-integration-check
        PLUGIN_TARGET  miopen_plugin                       # this provider's plugin .so CMake target
        ENGINE_NAME    MIOPEN_ENGINE                       # --test-engine
        INSTALL_SUBDIR miopen_plugin
        TEST_CONFIG    ${CMAKE_CURRENT_SOURCE_DIR}/config/MIOPEN_ENGINE.toml
        TEST_CATEGORIES_YAML ${MIOPENPROVIDER_INTEGRATION_CATEGORIES_YAML}
    )
endif()
```

This produces the resolved invocation:

```
hipdnn_integration_tests --test-article <plugin.so> --test-engine <ENGINE> [--test-config <toml>] [--gtest_filter=...]
```

When `TEST_CATEGORIES_YAML` is supplied, the helper additionally creates
tier-labelled CTest suites (via the shared `apply_test_category_labels()`
from `shared/ctest/TestCategories.cmake`) so `ctest -L quick|standard|...`
selects tiers for the external cross-provider run the same way it does for
native tests — this is what makes `ctest -L quick` work uniformly across
`hipdnn-check`, `miopen-provider-check`, and
`miopen-provider-external-integration-check`.

`hip-kernel-provider` does **not** register `-check` targets the same way —
its tests are staged through its install bucket rather than this shared
test-target machinery, so don't expect an
`hip-kernel-provider-external-integration-check` target to exist by the same
pattern as miopen/hipblaslt.

## 6. Running the binary directly against an engine

```bash
# Standalone — point at a specific plugin and pin the engine
./bin/hipdnn_integration_tests \
    --test-article /path/to/libmiopen_plugin.so \
    --test-engine  MIOPEN_ENGINE \
    --test-config  /path/to/MIOPEN_ENGINE.toml \
    --gtest_filter='quick_*'

# Superbuild — plugin discovery is automatic (loads whatever is installed)
./bin/hipdnn_integration_tests

# C++ tests only, skip bundle/sweep discovery
./bin/hipdnn_integration_tests --no-bundles
```

Flags (see `src/main.cpp`'s argparse block for the authoritative list):

| Flag | Purpose |
|---|---|
| `--ta`, `--test-article <path>` | Path to the engine plugin `.so`/`.dll`. Omit to use hipDNN's default plugin discovery. |
| `--te`, `--test-engine <name>` | Pin the run to one engine (e.g. `MIOPEN_ENGINE`). **Always pass this when validating a single provider** — see §7.4. |
| `--tc`, `--test-config <toml>` | Per-engine tolerance/skip TOML — §4. |
| `--reference-executor cpu\|gpu` | Which reference implementation validates C++ parameterized (non-bundle) tests. Also `HIPDNN_TEST_REFERENCE_EXECUTOR`. |
| `--vm`, `--verification-mode auto\|golden\|gpu\|cpu\|golden-check` | How **bundle** output is verified (independent of `--reference-executor`). `auto` tries golden → GPU ref → CPU ref → skip, in that order. Also `HIPDNN_TEST_VERIFICATION_MODE`. |
| `--no-bundles` | Disable bundle/sweep registration, leaving only compiled-in C++ tests. Also `HIPDNN_TEST_ALLOW_BUNDLES=0`. |
| `--gd`, `--golden-data-dir <path>` | Bundle data root. Defaults to `<exe>/../lib/integration-test-bundles/`. Also `HIPDNN_TEST_GOLDEN_DATA_DIR`. |
| `--fail-on-unsupported` | FAIL instead of SKIP when no engine supports a graph (off by default — default behavior is SKIP; see §7.2). |
| `--skip-graph-validation` | PASS immediately after confirming engine support, without executing/validating the graph. |
| `--generate-support-matrix [file]` | Emit a markdown support matrix (default `support_matrix.md`) summarizing which engines accepted which graphs during this run. Use this to diff coverage across runs — see §7.2. |
| `--capture-bundles <dir>` | Dump compiled-in C++ graph tests as JSON bundles (migration tooling, not day-to-day). |
| `--gtest_filter=<pattern>` | Standard GTest filter, passed through after hipDNN's own args are parsed. |

Via CTest, from a provider's build/install dir:

```bash
cmake --build build --target miopen-provider-external-integration-check   # exact CI invocation
ctest -L quick                                                            # tier-filtered, once built once
```

## 7. Failure modes that let a broken suite look green

These are the footguns to actively check for — an agent that only reads
"ctest passed" / exit code 0 can miss all of them.

### 7.1 Zero tests run is a FAILURE, not a clean pass

`main.cpp` runs `RUN_ALL_TESTS()` and then checks
`UnitTest::test_to_run_count()`. **If it is 0, and either `--test-engine`
was supplied or the bundle data directory exists**, the binary prints an
explicit diagnostic and returns 1:

```
Error: zero tests ran.
  registered:      <N> test(s) in <M> suite(s)
  selected:        0 (nothing matched --gtest_filter)
  gtest_filter:    <filter>
  bundle data dir: <path> (exists|MISSING)
  registered suite: ...
```

Treat this message as **always a configuration bug**, never an
infrastructure fluke: wrong `--test-article` path, a typo'd
`--test-engine` name, a `--gtest_filter` that matches nothing, or a plugin
that failed to load (check the `registered suite:` list against what you
expected — 0 registered suites at all usually means the plugin never
loaded). The "registered N / selected 0" split is intentional: it tells you
whether the problem is discovery (N=0) or filtering (N>0, selected=0) —
those have different fixes, use the numbers, don't guess.

**Gap to know about:** a run with *neither* `--test-engine` nor an existing
bundle data dir (e.g. a bare hipDNN-only checkout with no provider
installed) is explicitly allowed to run empty — the guard does not fire.
So absence of this error is not proof of a non-empty run in every context;
when you have `--test-engine` or bundle data, expect the guard to catch you,
but always sanity-check the "TEST COVERAGE SUMMARY" (`Passed:`/`Skipped:`/
`Failed:` / total) that prints right after, regardless.

Also applies at the CTest layer, one level up: `ctest -L <tier>` or
`ctest -R <pattern>` that matches **zero** registered tests reports "No
tests were found" and a non-zero exit — a typo'd label/regex is a
misconfiguration, not "nothing to run."

### 7.2 Support claims exist as a schema, but nothing enforces them yet — a passing suite can hide a real regression

RFC 0015 (`docs/rfcs/0015_EngineSupportClaims.md`) defines a
`{Name}.support.json` (and template-sweep `support.json`) claim file: which
`(engine, arch, platform)` combinations a bundle's author asserts must stay
supported. Its stated purpose is exactly to close a **silent regression
channel**: today, when an engine that used to accept a graph starts
declining it, the golden-reference framework maps that to `GTEST_SKIP` —
the suite stays green and nobody is told coverage was lost.

**As of this writing, the enforcement half of RFC 0015 is not wired in.**
`src/harness/bundle/SupportClaims.{hpp,cpp}` implement the parser and
`SupportClaims::isClaimed()`/`SweepSupportClaims::isClaimed()`, but grep the
harness: nothing in `IntegrationBundleVerificationHarness.cpp` or
`main.cpp` calls `loadSupportClaims()`, `loadSweepSupportClaims()`, or
`isClaimed()` outside the parser's own unit test
(`tests/TestSupportClaims.cpp`). There is no `--write-support-claims` or
`--enforce-support-claims` flag in `main.cpp`'s argparse block. Concretely:
when an engine declines a graph, `get_ranked_engine_ids()` reports it
missing, the harness throws `EngineNotApplicableError`, and that is caught
and converted to an unconditional `GTEST_SKIP()` — regardless of whether a
`support.json` next to the bundle claims that engine as supported.

**Consequences for you:**
- A bare "ctest passed" / "0 failed" result is **not** evidence that an
  engine still supports what it supported yesterday. If you're
  investigating "did engine X regress on graph Y," don't stop at exit
  code — check the `Skipped: N` line in the "TEST COVERAGE SUMMARY", or
  run with `--generate-support-matrix` and diff the emitted markdown
  against a prior run.
- A `support.json`/`{Name}.support.json` existing next to a bundle today is
  informational/aspirational only — it is not yet a live gate. Don't tell a
  user "this is protected by a support claim" as if it fails CI; it doesn't,
  yet.
- Before relying on this section, re-check `main.cpp`'s argparse block and
  `IntegrationBundleVerificationHarness.cpp` for `--write-support-claims`/
  `--enforce-support-claims`/`isClaimed(` call sites — if RFC 0015's
  enforcement ladder has since landed, this caveat is stale and the claim
  files are load-bearing again.

### 7.3 Missing bundle data degrades to SKIP, not a build/config error

If a bundle's `.tensor*.bin` golden data hasn't been `dvc pull`ed, `auto`
verification mode falls through to GPU ref, then CPU ref, then skips —
quietly. Look for the "Bundle tests are enabled but …" warning in the log,
or force `--verification-mode gpu`/`cpu` to bypass golden data entirely and
confirm the graph itself still runs.

### 7.4 Omitting `--test-engine` lets hipDNN's own engine selection pick the winner

With multiple plugins loaded (a multi-provider superbuild) and no
`--test-engine`, a bundle that "passes" may have been validated against a
different engine than the one you think you're testing — hipDNN's normal
selection heuristic runs, not a pinned choice. Always pass `--test-engine`
when isolating one provider's behavior.

### 7.5 `test_skips` and `tolerance_overrides` resolve conflicts in opposite orders

See §4: skips are first-match-wins, tolerance overrides are last-match-wins.
Adding a new entry near an existing one without checking which list you're
in is an easy way to have your new rule silently overridden (or to silently
override an existing one).

### 7.6 `-DBUILD_CPP_GRAPH_TESTS=OFF` is the default

A "this C++ integration test doesn't run" report for a test under
`src/integration-tests/{op}/` is very often just this flag (default off; the
provider CI checks run bundles only) — confirm the build option before
treating it as a regression. One exception: `resample/` has no bundle
equivalent yet and stays always-built migration debt.

### 7.7 `GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST` can hide a fully-unsupported legacy C++ suite

Pre-bundle-era `INSTANTIATE_TEST_SUITE_P(..., BuildEngineTestMatrix<...>(...))`
calls carry this macro so that zero engines supporting a fixture doesn't
hard-fail GTest registration — but it also means an entire suite can
register 0 test cases without comment. Only relevant with
`BUILD_CPP_GRAPH_TESTS=ON`; bundles (§1) don't have this failure mode
because bundle registration counts are checked by the §7.1 guard instead.

## 8. Adding a new test — quick decision guide

1. **"Does this graph run and match a reference on an engine?"** → add/extend
   a bundle. Prefer a template-sweep over a single-graph bundle unless there
   is exactly one concrete graph with no axis to vary. Use
   `migration-scripts/import_graph.py --graph <file>.json --bundle-dir integration-test-bundles/`
   — it dedups by structure hash and appends or creates as needed; never
   hand-write a `sweep.json`.
2. **Anything else** (unhappy paths, API-contract behavior, serialization
   round-trips, benchmarking knobs, determinism) → C++ via
   `add_always_built_test_sources()` plus a
   `HIPDNN_IT_ALWAYS_BUILT_SOURCES` entry explaining why it can't be a
   bundle.
3. **Validating the reference executor itself** (not an engine) → C++ under
   `tests/gpu-ref/`, always defining all four tiers
   (Smoke/Standard/Comprehensive/Full instantiations).

## See also

- `hipdnn-superbuild-test` skill — discovers and runs the actual CMake/ctest
  targets in an existing superbuild (component/scope selection, Windows DLL
  PATH, the `<provider>-external-integration-check` reproduction). Use it to
  execute; use this skill to interpret the result.
- `dnn-providers/integration-tests/README.md` — the canonical, longer-form
  version of §1–§6 and §8, including bundle-format details and DVC workflow.
- `projects/hipdnn/docs/rfcs/0006_PluginAgnosticIntegrationTests.md` —
  why the suite is plugin-agnostic and how CI time budgets are managed.
- `projects/hipdnn/docs/rfcs/0011_GoldenReferenceValidation.md` — bundle vs
  template-sweep on-disk format.
- `projects/hipdnn/docs/rfcs/0015_EngineSupportClaims.md` — the support
  claim schema referenced in §7.2, including its intended enforcement
  ladder once implemented.
