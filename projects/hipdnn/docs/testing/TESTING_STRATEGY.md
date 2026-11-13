# hipDNN Testing Strategy

Owner: @ROCm/hipdnn-reviewers | Last Updated: 2026-10-22

[Back to Testing](../TESTING.md)

hipDNN is tested in four layers: cheap unit tests everywhere, mock-based API tests for the core, the same quality bar for every provider, and one shared integration project that runs against every provider. This page covers why and what each layer owns. For commands, tiers and CI placement, see [Testing](../TESTING.md). For current limits, see [Known Testing Gaps](./KNOWN_TESTING_GAPS.md).

## Methodology

| Layer | Approach | Goal |
|---|---|---|
| Code units | Fast unit tests with high coverage | Cover every unit that can be tested cheaply |
| API surface and contract | Integration tests with test plugins and mocks: frontend to backend, backend to plugin | Stop issues leaking to users of hipDNN core |
| Providers | The same bar for each: unit tests on everything possible, mocks for dependencies, high coverage | Uniform provider quality |
| Real integrations | One shared integration project run against every provider | A provider's regression or unique case expands coverage for all providers |

Three terms used on this page:

- **Bundle**: a serialized graph plus its sweep cases, run by the shared integration binary. See [Integration test strategy](#integration-test-strategy).
- **Oracle**: the reference a result is graded against (CPU reference, GPU reference, or checked-in golden data).
- **Enablement**: a test that shows an op, engine or feature works at all on a provider, as opposed to a test that stresses it.

### Ownership

| Owner | Validates |
|---|---|
| hipDNN core | Public APIs, graph data, routing, plugin lifetime, frontend/backend flow |
| Shared integration project | Shared graph cases, oracles, tolerances, the provider-agnostic harness |
| Providers | Graph translation, support checks, engine selection, provider skips and tolerances, operation results on supported GPUs |
| Lower-level libraries and kernels | Kernel selection, compilation, workspace use, device execution |
| CI and lab | Runners, hardware, system software, repeatable environments |

A green core job proves that its selected routing and API tests passed. It does not prove that every provider, op, shape, datatype, engine or GPU works. Provider support comes from provider documentation and runtime support checks. [CODEOWNERS](../../../../.github/CODEOWNERS) routes reviews; it does not define support.

## Test types

| Type | Where | What it checks | Runs in CI |
|---|---|---|---|
| Unit | `backend/tests/`, `frontend/tests/`, `data_sdk/tests/`, `flatbuffers_sdk/tests/`, `plugin_sdk/tests/`, `test_sdk/tests/` under `projects/hipdnn/` | One component's internal behavior; mostly CPU-only | TheRock `hipdnn` component, every PR and push that touches hipDNN |
| Frontend/backend API | `projects/hipdnn/tests/backend/`, `tests/frontend/`, `tests/test_plugins/`, `tests/backend_logging/` | Public API, descriptor conversion, routing, plugin management, against controlled test plugins; safe backend shutdown while log callbacks run. Says nothing about provider numerics | TheRock `hipdnn` component |
| Provider unit | `dnn-providers/<name>/tests/` (`dnn-providers/hip-kernel-provider/src/tests/` for HIP-kernel) | One provider's translation, support checks, engine configuration, errors | TheRock provider components |
| Shared provider integration | `dnn-providers/integration-tests/` | Bundles run through one provider and graded against an oracle | TheRock `hipdnn-integration-tests` and provider components |
| C++ integration (special cases) | `dnn-providers/miopen-provider/integration_tests/`, `dnn-providers/hipblaslt-provider/integration_tests/`, `dnn-providers/hip-kernel-provider/src/integration_tests/` | Behavior specific to one provider, such as determinism and tuning | TheRock provider components |
| Samples | `projects/hipdnn/samples/`, `projects/hipdnn/tests/cudnn_samples/` | Documented usage compiles and runs | `hipdnn-samples`, `hipdnn-cudnn-samples` |
| Install and consumption | `hipdnn_install` component | An installed tree can be consumed by a downstream project | TheRock `hipdnn_install` |
| Sanitizers | ASAN and TSAN builds | Memory and thread safety of the code that ran | [Sanitizers](#sanitizers) |
| Static analysis | clang-tidy | Style and common C++/HIP errors | [Static analysis](#static-analysis) |
| Coverage | `-DHIPDNN_ENABLE_COVERAGE=ON` | Which code ran | Reported on each PR |
| Performance | dnn-benchmarking | Engine timing across workloads | [Performance](#performance) |

Every GPU-dependent test calls `SKIP_IF_NO_DEVICES()` so the suite runs on CPU-only machines. A skip is not a pass for that GPU configuration. Add a regression test for each bug fix, beside the component's existing tests, using existing fakes and mocks for outside dependencies. Test names follow [Coding Style and Naming Guidelines](../CodingStyleAndNamingGuidelines.md).

## Integration test strategy

One shared test project, `dnn-providers/integration-tests/`, builds a graph, runs it on a provider's engine, and checks the result against a reference. Every provider (MIOpen, hipBLASLt, hip-kernel) runs the same provider-facing binary, `hipdnn_integration_tests`, against its own plugin, so a case added for one provider tests all of them. The [integration test documentation](../../../../dnn-providers/integration-tests/docs/README.md) is the reference for flags, formats and authoring; run commands are in [Testing](../TESTING.md#building-and-running-tests).

- **Engine-pinned lanes.** `--test-engine` pins a run to one engine. An op that engine does not support skips instead of falling through to another loaded engine.
- **Bundles.** A bundle is a graph stored as JSON plus a case matrix of shapes, datatypes and layouts. A bundle may also carry golden tensors. New graph-verification tests are bundles: adding a case is a data edit, not a recompile. Bundles live in `dnn-providers/integration-tests/integration-test-bundles/` and are what CI runs; the legacy C++ graph tests build only with `-DBUILD_CPP_GRAPH_TESTS=ON`, which is off by default.
- **Oracles.** `--verification-mode` selects the oracle: golden data, GPU reference or CPU reference, or `auto`, which tries them in that order and fails if none can verify. Modes: [Verification modes](../../../../dnn-providers/integration-tests/docs/running-tests.md#verification-modes).
- **Support claims.** A bundle may carry a support sidecar (`<graph>.support.json`, or `support.json` for a sweep) that promises a named engine supports that graph on an arch and platform. A broken claim fails the test instead of skipping. See [support-claim-enforcement.md](../../../../dnn-providers/integration-tests/docs/support-claim-enforcement.md).
- **Tiers.** The tier of a bundle is its directory: `quick/`, `standard/`, `comprehensive/`, `full/`. A higher tier includes the lower ones. Tier meaning, budgets and CI placement: [Test tiers](../TESTING.md#test-tiers).
- **Per-provider category filtering.** Each provider keeps a `test_categories_integration.yaml` (hip-kernel keeps one per engine, named `<ENGINE>_test_categories_integration.yaml`) that selects which cases run per tier for the shared binary. Lower tiers are scoped by bundle and name-prefix patterns. `full` is `*` for general-purpose engines and is limited to SDPA cases for attention-specific engines. A provider that implements a subset of ops relies on the harness's engine-support check, which skips ops the engine does not support. See [Test tiers](../../../../dnn-providers/integration-tests/docs/running-tests.md#test-tiers).
- **Golden data checks.** `hipdnn_golden_data_tests` recomputes golden tensors with the reference executors. It validates our data, not a provider.
- **One shared set covers every provider.** A provider-specific regression or unique graph-run case goes in the shared bundle set, not in a provider-local C++ test. Every other provider then runs it too. An engine that cannot run the case skips it, or fails its support claim if it promised to. Provider-local C++ tests are for behavior that is not "a graph runs and matches a reference", such as error paths or plugin-specific behavior.
- **Adding a provider.** A provider joins with a plugin, a test target, a `test_categories_integration.yaml` and a `config/<ENGINE>.toml`; wiring: [Add an engine lane](../../../../dnn-providers/integration-tests/docs/adding-tests.md#add-an-engine-lane). To start, add the target, run `ctest -L '^quick$'`, and add `.support.json` sidecars for the graphs the engine promises to run. Skips are data in the engine's `config/<ENGINE>.toml`, each with a `reason`, and need a tracked issue ([Flaky tests and skips](#flaky-tests-and-skips)).

Today, bundles check that a graph runs on an engine and matches an oracle, with thin per-op depth. The direction is a unified gate that gives high confidence in the quality of a new integration or kernel source, built from detailed per-op coverage. See [Roadmap: Testing](../Roadmap.md#testing-1).

## Tier policy

Tier names, roles, budgets and what runs in CI today are in the [tier table](../TESTING.md#test-tiers). The principles:

- Pre-submit stays tightly costed: very quick CPU-only unit tests that give the most signal, plus quick feature coverage using pairwise feature coverage per op. The goal is a minimal enablement set that flags obvious breaks across providers.
- The next tier adds slower enablement tests and high-priority regression cases.
- `full` is much broader cases and regression testing on a release cadence. See [Release Testing](./RELEASE_TESTING.md).

## Flaky tests and skips

- Every skip and every tolerance override needs a tracking issue, and the code names it (`[[test_skips]]` `reason` in provider config, a comment beside `SKIP_IF_*`).
- A flaky or inconsistent test is logged as an issue and fixed. It is not left as a non-blocking failure.
- Any workaround that disables behavior (a skip, a suppression, a `DISABLED_` test) has a tracked issue and is removed when the issue closes.
- Do not disable or skip a test only to make a check pass.

## Sanitizers

Run sanitizers for changes that touch risky code:

- ASAN: memory ownership, allocation, object lifetime, serialization, plugin loading, cleanup after failures.
- TSAN: threads, shared state, callbacks, logging, synchronization.

A nightly workflow builds `develop` with device ASAN for gfx94X and gfx950. The team runs ASAN tests itself every week on gfx942. TSAN is supported for core on Linux; no CI job in this repo runs it, and providers have no TSAN option. ASAN and TSAN use separate builds and cannot be enabled together. A test that cannot run under ASAN skips through `SKIP_IF_ASAN()` or is left out of the ASAN build. A known upstream error is suppressed in [`AsanDefaultSuppressions.cpp`](../../test_sdk/src/AsanDefaultSuppressions.cpp) instead, so its tests still run.

A sanitizer run shows that the code which executed has no memory or thread error. It does not prove numerical correctness, and skipped tests were not checked. Commands: [Testing](../TESTING.md#sanitizers-and-static-analysis) and [Building](../Building.md#address-sanitizer-build).

## Static analysis

clang-tidy runs in the Linux superbuild CI and flags style and common errors. It is on by default for Linux builds and off for Windows builds. See [Building: Clang Tools](../Building.md#clang-tools).

## Coverage

Code coverage is reported on every PR and must be 80% or more. It is a review criterion, not a merge block. Coverage shows which code ran; it does not show correct behavior or GPU support. To check locally, build with `-DHIPDNN_ENABLE_COVERAGE=ON` and run the `unit-coverage` target for a unit-only report; `coverage` runs every test. Reports go to `coverage-report/`.

## Performance

Use [dnn-benchmarking](https://github.com/ROCm/dnn-benchmarking#readme) for graph benchmarks. It runs manually today. Release use: [Release Testing](./RELEASE_TESTING.md#8-run-performance-benchmarks).

Record with each result: provider, engine, GPU, workload revision, and software build. Then:

- Compare only against a baseline from the same GPU architecture, on the same system setup when possible.
- Keep GPU graph time separate from host submission time.
- Re-run a suspicious result before calling it a regression.
- Do not compare performance when correctness tests fail.
