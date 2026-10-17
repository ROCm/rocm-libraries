# hipDNN Testing Strategy

Owner: @ROCm/hipdnn-core | Last Updated: 2026-10-15

hipDNN is tested in four layers: cheap unit tests everywhere, mock-based API tests for the core, the same quality bar for every provider, and one shared integration project that runs against every provider. This page covers why and what each layer owns. For commands, tiers and CI placement, see [Testing](../TESTING.md). For current limits, see [Known Testing Gaps](./KNOWN_TESTING_GAPS.md).

## Methodology

| Layer | Approach | Goal |
|---|---|---|
| Code units | Fast unit tests with high coverage | Cover every unit that can be tested cheaply |
| API surface and contract | Integration tests with test plugins and mocks: frontend to backend, backend to plugin | Stop issues leaking to users of hipDNN core |
| Providers | The same bar for each: unit tests on everything possible, mocks for dependencies, high coverage | Uniform provider quality |
| Real integrations | One shared integration project run against every provider | A provider's regression or unique case expands coverage for all providers |

Three terms used on this page:

- **Bundle**: a serialized graph plus its sweep cases, run by the shared integration binary. See [Integration Testing](./INTEGRATION_TESTING.md).
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
| Frontend/backend API | `projects/hipdnn/tests/backend/`, `tests/frontend/`, `tests/test_plugins/` | Public API, descriptor conversion, routing, plugin management, against controlled test plugins. Says nothing about provider numerics | TheRock `hipdnn` component |
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

## Tier policy

Tier names, roles, budgets and what runs in CI today are in the [tier table](../TESTING.md#test-tiers). The principles:

- Pre-submit stays tightly costed: very quick CPU-only unit tests that give the most signal, plus quick feature coverage using pairwise feature coverage per op. The goal is a minimal enablement set that flags obvious breaks across providers.
- The next tier adds slower enablement tests and high-priority regression cases.
- `full` is much broader cases and regression testing on a release cadence. See [Release Testing](./RELEASE_TESTING.md).
- A CI test step is limited to 30 minutes per shard. Tiers that need longer run past that limit by sharding the component across jobs.

## Flaky tests and skips

- Every skip and every tolerance override needs a tracking issue, and the code names it (`[[test_skips]]` `reason` in provider config, a comment beside `SKIP_IF_*`).
- A flaky or inconsistent test is logged as an issue and fixed. It is not left as a non-blocking failure.
- Any workaround that disables behavior (a skip, a suppression, a `DISABLED_` test) has a tracked issue and is removed when the issue closes.
- Do not disable or skip a test only to make a check pass.

## Performance

Use [dnn-benchmarking](https://github.com/ROCm/dnn-benchmarking#readme) for graph benchmarks. It runs manually today.

Record with each result: provider, engine, GPU, workload revision, and software build. Then:

- Compare only against a baseline from the same GPU architecture, on the same system setup when possible.
- Keep GPU graph time separate from host submission time.
- Re-run a suspicious result before calling it a regression.
- Do not compare performance when correctness tests fail.

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
