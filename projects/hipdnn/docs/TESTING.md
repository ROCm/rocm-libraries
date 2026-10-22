# hipDNN Testing

Owner: @ROCm/hipdnn-reviewers | Last Updated: 2026-10-22

## Component overview

This page shows how to build and run hipDNN tests, what each CI tier runs, and what to run before you push.

Core-only change, from `projects/hipdnn/`:

```bash
cmake --preset release
cmake --build build/release
ctest --test-dir build/release -L '^standard$' --output-on-failure
```

Provider or cross-project change, from the repository root:

```bash
cmake --preset hipdnn-providers -DROCM_LIBS_ENABLE_ROOT_CTEST=ON
cmake --build build
ctest --test-dir build -L '^standard$' --output-on-failure
```

`ctest` does not build. Build first. Anchor tier labels (`'^standard$'`): `-L standard` is a regex and also matches any label that contains the word. Replace `standard` with `quick`, `comprehensive` or `full` to run another tier.

### Testing documentation

- [Testing Strategy](./testing/TESTING_STRATEGY.md): methodology, layers, integration test strategy, and tier, flaky-test, sanitizer, coverage and performance policy.
- [Known Testing Gaps](./testing/KNOWN_TESTING_GAPS.md): what a green run does not prove.
- [Release Testing](./testing/RELEASE_TESTING.md): the release verification procedure. [Release Run Template](./testing/RELEASE_RUN_TEMPLATE.md): the record to fill in while following it.
- [Integration test documentation](../../../dnn-providers/integration-tests/docs/README.md): the shared provider test project (flags, formats, authoring).
- [Building hipDNN](./Building.md): prerequisites, presets, targets. [Coding Style and Naming Guidelines](./CodingStyleAndNamingGuidelines.md) covers test names.

## Development workflow

### Building and running tests

Build options and presets: [Building hipDNN](./Building.md). Superbuild presets include `hipdnn`, `miopen-provider`, `hipblaslt-provider`, `hip-kernel-provider`, `hipdnn-providers`, `hipdnn-providers-all`, `hipdnn-samples`, and `hipdnn-dev-all`.

**Standalone core** (above). Use `debug` instead of `release` for a debugger build. To run one test, use `ctest -R <regex>` or the test binary with `--gtest_filter`.

**Superbuild.** Root `ctest` sees tests only when configured with `-DROCM_LIBS_ENABLE_ROOT_CTEST=ON`. Superbuild means one CMake build of hipDNN, the providers and the samples from the repository root. Pick the smallest preset that includes the affected provider (`hipdnn-providers` covers MIOpen and hipBLASLt; `hipdnn-providers-all` adds HIP-kernel). Tier targets for hipDNN core do not need that flag:

```bash
cmake --build build --target hipdnn-standard-check
```

`hipdnn-<category>-check` and `hipdnn-<category>-check-verbose` exist for each category in [`test_categories.yaml`](../test_categories.yaml): the tiers `quick`, `standard`, `comprehensive` and `full`, plus `ffm-quick`, `unit` and `integration`. They build and run the hipDNN core tests only; `hipdnn-check` runs all core tests. Provider and integration-tests targets are listed in [Building hipDNN: Superbuild Targets](./Building.md#superbuild-targets).

**One provider.** Build the provider's preset (for example `miopen-provider`), then run the shared integration suite against it:

```bash
cmake --build build --target miopen-provider-external-integration-check
```

**Installed tree, as TheRock runs it.** CI installs the build, then runs ctest inside each component's directory in the install:

```bash
ctest --test-dir <install>/bin/<dir> -L '^standard$' -LE ex_gpu --output-on-failure --parallel 1 --timeout 7200
```

`<dir>` is `hipdnn`, `hipdnn_integration_tests_ctest`, `hipdnn_samples`, `miopen_plugin`, `hipblaslt_plugin`, or `hip_kernel_provider`. When the component has suites labelled for the runner's GPU architecture (for example `miopen_plugin` on gfx110X), CI uses `-L '^ex_gpu_<arch>$'` in place of `-LE ex_gpu`.

**Integration binary.** Run it directly to slice a run. In a superbuild it finds the plugin itself; standalone, name the plugin:

```bash
./bin/hipdnn_integration_tests --test-article /path/to/libmiopen_plugin.so --test-engine <ENGINE> --gtest_filter='quick_*'
```

- `--test-engine` pins the run to one engine. `--test-article` is the plugin `.so`.
- Bundle tests (data-driven graph cases) run by default; `--no-bundles` runs only the C++ tests compiled into the binary.
- `--verification-mode auto|golden|gpu|cpu` selects the oracle (see [Verification modes](../../../dnn-providers/integration-tests/docs/running-tests.md#verification-modes)). `gpu` and `cpu` need no golden-data pull.
- Bundle suites are named `{tier}_{Op}_{Topology}`, so `--gtest_filter='*_Batchnorm_*'` selects one op. More flags: [command-line reference](../../../dnn-providers/integration-tests/docs/running-tests.md#command-line-reference--hipdnn_integration_tests).

**Samples.** Configure the `hipdnn-samples` preset with `-DROCM_LIBS_ENABLE_ROOT_CTEST=ON`; it builds hipDNN, the providers, and the samples together. Then run `ctest --test-dir build -L '^standard$'`. Standalone sample builds: [Building hipDNN: Building the Samples](./Building.md#building-the-samples).

**Without a GPU.** Tests that need a device call `SKIP_IF_NO_DEVICES()` and skip. `hipdnn_integration_tests` prints "No HIP devices available" and exits 0. A GPU-less run proves the CPU tests only.

**Windows.** The same `ctest` commands apply from a Windows build environment; see [Building hipDNN: Windows](./Building.md#windows) for setup.

### Reading results

- **ctest:** add `--output-on-failure` to print the log of each failing test. Add `--output-junit results.xml` for a JUnit file.
- **Integration binary:** prints `REFERENCE EXECUTOR ERRORS` and `UNVERIFIABLE BUNDLES` lists when non-empty, then a `TEST COVERAGE SUMMARY` that ends with a `Verified by: golden N, gpu_ref N, cpu_ref N, none N` line (what each bundle was checked against). `none` means nothing was compared.
- **CI logs:** each component is its own step in the `therock-multi-arch-ci.yml` run, named by component (`hipdnn`, `hipdnn-integration-tests`, `miopenprovider`, and so on).

### Before you push

- Build, then run `ctest -L '^standard$'` for each component you changed.
- Run the affected provider's tests for any change to plugin loading, graph execution, operation support, or provider-facing behavior.
- Test on the hardware your change affects. A provider change must be validated on the GPUs it targets; PR CI covers only a few ([Known Testing Gaps](./testing/KNOWN_TESTING_GAPS.md#hardware-coverage)).
- Add a test that fails before the fix and passes after it for every bug fix.
- Run the sanitizer that matches the risk: ASAN for memory ownership and lifetime; TSAN for threads and shared state.
- Format and static analysis: `hipdnn-check-format`, and `hipdnn-tidy` on Linux.
- Do not disable a test to get a check green. Track every skip or tolerance override in an issue ([policy](./testing/TESTING_STRATEGY.md#flaky-tests-and-skips)).
- List the commands you ran in the PR description. Do not mark unrun checks as passed.

## Pre-submit / CI gates

### Test tiers

Every test belongs to a tier, expressed as a ctest label: `quick`, `standard`, `comprehensive`, `full`. Tiers are cumulative in intent; in practice, how much each tier selects differs by project, and core and samples run all their tests at every tier because they are fast. Budgets and tier policy follow TheRock (ROCm's multi-repository build and CI system): [test_filtering.md](https://github.com/ROCm/TheRock/blob/main/docs/development/test_filtering.md). Principles: [Testing Strategy: Tier policy](./testing/TESTING_STRATEGY.md#tier-policy). Standalone and superbuild builds need third-party libraries from the environment or `-DALLOW_FETCH_DEPS=ON` ([Building hipDNN](./Building.md#third-party-libraries)). CI facts below are from this repo's workflows, checked 2026-10-22.

| Tier | Role (intended) | Runs today in CI | Budget |
|------|-----------------|------------------|--------|
| `quick` | Pre-submit minimum: CPU-only unit tests plus a minimal enablement set that flags obvious breaks across providers | Workflow dispatch; fallback when a PR or push cannot be mapped to projects (shared-CI script or config changes, unclassified paths) | < 5 min |
| `standard` | `quick` plus slower enablement and high-priority regression cases | Every PR and push to `develop` that touches hipDNN: Linux gfx94X, Windows gfx110X | < 30 min |
| `comprehensive` | Nightly breadth | Nightly: Linux gfx94X, Windows gfx1151 | < 2 h |
| `full` | Release: much broader cases and regression testing | TheRock prerelease builds (outside this repo) | 2 h or more |

- Each CI test step is limited to 30 minutes per shard (10 for `hipdnn-cudnn-samples`). A tier that runs longer reaches its budget by sharding: the component splits its tests across jobs, each within the 30-minute limit. Only `hipkernelprovider` shards today (2 shards, except `quick`).
- `hipdnn_install` and `hipdnn-cudnn-samples` ignore the tier and run unfiltered `ctest`. They run on run-all triggers only.
- The superbuild CI applies no tier filter, so it runs the suites of every tier ([gates](#what-gates-a-pull-request)).
- GPUs that PR and nightly CI do not cover: [Known Testing Gaps](./testing/KNOWN_TESTING_GAPS.md#hardware-coverage).

### Pull request labels

Labels on a PR change what CI runs. Required GitHub checks are configured outside this repo.

| Label | Effect |
|-------|--------|
| `test:<component>` (for example `test:hipdnn`) | Restricts the test components to the listed ones |
| `ci:skip` | Skips CI |
| `ci:asan`, `ci:host-asan` | Enables the PR ASAN workflow, which builds ASAN (`ci:asan`) or host-ASAN (`ci:host-asan`) binaries and does not run tests |
| `ci:gfx*`, `ci:build:gfx*`, `ci:test:gfx*`, `ci:run-all-archs`, `ci:platform:linux`, `ci:platform:windows` | Selects GPU families and platforms |
| `ci:cpu-test-only` | Disables the GPU test runner for all families |
| `test_filter:<tier>` | No effect on the PR workflow: its tier is `standard` (or `quick` on a fallback) |

### What gates a pull request

| Check | What it runs |
|-------|--------------|
| `hipdnn-superbuild-ci.yml` | Superbuild on Linux (gfx942 target) and Windows (gfx1151 target), then `ctest --test-dir build --output-on-failure --no-tests=error`, which produces no JUnit file. Triggers on PRs that touch `projects/hipdnn/**` or `dnn-providers/**`, and nightly. Skips the builds when every changed file is documentation (`*.md`, `*.rst`, `docs/**`); the skipped job counts as passing |
| `therock-multi-arch-ci.yml` | `standard` tier on Linux gfx94X and Windows gfx110X for the affected components |
| `therock-multi-arch-ci-asan.yml` | ASAN build (host-ASAN with `ci:host-asan`), only with `ci:asan` or `ci:host-asan`; no tests |

Workflow files are the source of truth; which checks are required is set in repository settings.

### Emulating CI

Climb only as far as you need.

1. **Run the label locally.** Build the preset for the affected component and run `ctest -L '^standard$'` as in [Building and running tests](#building-and-running-tests). This matches what a PR runs, minus the GPU architecture.
2. **Run the installed tree.** Install the build and run the `ctest --test-dir <install>/bin/<dir> ...` command above. This catches missing install rules and files that only exist in the build tree.
3. **Dispatch the workflow.** Run `therock-multi-arch-ci.yml` through `workflow_dispatch` with `linux_amdgpu_families` or `windows_amdgpu_families` (for example `gfx950`) and `pr_labels: test:hipdnn`. Dispatch runs `quick`.

Local and CI runs differ in:

- GPU architecture and driver. CI PR runs use gfx94X on Linux and gfx110X on Windows; your GPU may differ.
- Tier selection. CI selects the tier from the trigger, not from the preset.
- Parallelism. CI runs `--parallel 1`.

When a CI result differs from your local run, compare the GPU architecture, tier and parallelism first. Keep the CI job log.

## Sanitizers and static analysis

Policy and CI coverage: [Testing Strategy: Sanitizers](./testing/TESTING_STRATEGY.md#sanitizers), [Static analysis](./testing/TESTING_STRATEGY.md#static-analysis) and [Coverage](./testing/TESTING_STRATEGY.md#coverage). Platform notes: [Building hipDNN: Address Sanitizer Build](./Building.md#address-sanitizer-build) and [Clang Tools](./Building.md#clang-tools).

```bash
# ASAN (standalone)
cmake --preset release -DBUILD_ADDRESS_SANITIZER=ON
cmake --build build/release
ctest --test-dir build/release -L '^standard$'

# TSAN (Linux host code; cannot be combined with ASAN)
cmake --preset release -DBUILD_THREAD_SANITIZER=ON

# clang-tidy
cmake --preset release -DENABLE_CLANG_TIDY=ON
cmake --build build/release --target hipdnn-tidy

# Coverage (reports in build/release/coverage-report/; other targets: unit-coverage, integration-coverage, current-coverage)
cmake --preset release -DHIPDNN_ENABLE_COVERAGE=ON
cmake --build build/release --target coverage
```

## Release testing

The hipDNN team runs the `full` tier and the benchmark corpus by hand before release.
Procedure and run record: [Release Testing](./testing/RELEASE_TESTING.md).
