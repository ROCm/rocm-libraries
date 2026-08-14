# hipDNN Testing Strategy

- **Owner:** `@ROCm/hipdnn-core` (see [CODEOWNERS](../../../.github/CODEOWNERS))

> **In a hurry?** Read [The short version](#the-short-version), then use
> [Choosing the Right Test Type](#choosing-the-right-test-type) before writing a test or
> [Known Risks and Gaps](#known-risks-and-gaps) before deciding how much confidence to place in a
> green check.

This document describes how hipDNN is tested today, which validation runs automatically, where
validation ownership changes hands, and what is not covered. It is the strategy and governance layer.
Detailed commands, test-authoring conventions, layer reference, release procedures, and the
run-record template are consolidated below.

This is a current-state document, not a statement that every desired control exists. A merged change
is present on `develop`; merge status alone does not prove that every supported configuration,
sanitizer, performance workload, or release requirement was validated.

## The short version

**Where confidence comes from.** Fast unit test suites exercise each hipDNN layer and the public API
with in-tree fake plugins. Frontend-to-backend integration tests verify graph construction, routing,
and execution flow. Real-provider and cross-provider suites run graphs on GPUs and compare results
against a reference executor. Providers, not the routing library, own per-ASIC numerical correctness.

**What runs on pull requests.** Applicable hipDNN and provider changes trigger the Linux and Windows
[hipDNN Superbuild CI](../../../.github/workflows/hipdnn-superbuild-ci.yml). That gate runs without a
GPU: architecture values select build artifacts/targets, while GPU tests skip. The primary
[TheRock CI](../../../.github/workflows/therock-ci.yml) supplies separate on-device coverage where its
live package matrix selects hipDNN, and runs its `standard` wrapper unless paths or labels cause it
to skip. [Pre-commit](../../../.github/workflows/pre-commit.yml) and
the [Libraries PR Bot](../../../.github/workflows/libraries-pr-bot.yml) also run. Checked-in workflow
files show when jobs run; they do not expose repository branch-protection settings, so this document
does not claim that every listed status is a required merge check.

**What is not automated.** hipDNN does not enforce the 80% code-coverage target and has no checked-in
Codecov upload or threshold. `ROCm/dnn-benchmarking` uses the hipDNN Python bindings to run graphs
defined in JSON. Performance testing is manual and has no set cadence today; weekly regression runs
are planned. The trusted ASAN check is also manual. Pull-request sanitizer jobs build but do not run
tests, and scheduled ASAN infrastructure is not yet an owned hipDNN signal.

**The biggest gaps.** No automated performance signal; pre-submit covers only the quick/smoke subset
of intended validation; tolerance overrides, skips, and other test workarounds lack one consistent
ownership and expiry policy; ASAN evidence is manually produced; coverage has no published baseline
or floor; and exact required-check settings are not visible in this repository.

## Contents

- [Component Overview](#component-overview)
- [Development Workflow](#development-workflow)
- [Developer Testing Commands and Authoring Reference](#developer-testing-commands-and-authoring-reference)
- [Testing Strategy and Layers](#testing-strategy-and-layers)
  - [Unit Test Strategy](#unit-test-strategy)
  - [Integration Test Strategy](#integration-test-strategy)
  - [Performance and Benchmarking](#performance-and-benchmarking)
- [Pre-submit and CI Gates](#pre-submit-and-ci-gates)
- [Coverage](#coverage)
- [Nightly Validation](#nightly-validation)
- [Supported and Validated Configurations](#supported-and-validated-configurations)
- [ASAN, TSAN, and Sanitizer Coverage](#asan-tsan-and-sanitizer-coverage)
- [Static Analysis](#static-analysis)
- [For New Contributors](#for-new-contributors)
- [Key Quality Concerns](#key-quality-concerns)
- [Release Validation](#release-validation)
- [Dependencies and Validation Handoffs](#dependencies-and-validation-handoffs)
- [Improvement Roadmap](#improvement-roadmap)
- [Known Risks and Gaps](#known-risks-and-gaps)
- [Detailed Testing Layer Reference](#detailed-testing-layer-reference)
- [Release and Milestone Test Plan](#release-and-milestone-test-plan)
- [Test Run Record Template](#test-run-record-template)
- [Owners and Review Cadence](#owners-and-review-cadence)

## Component Overview

The project [README](../README.md#overview) owns the product overview and
[component inventory](../README.md#project-structure). The [Design Overview](Design.md) owns
architecture and component-boundary details.

Only the testing-specific ownership split is repeated here: hipDNN validates its public APIs,
graph/data routing, and provider boundary; provider projects and the shared cross-provider suite
validate numerical correctness on their supported GPU architectures. That split determines which
test layer owns a change.

## Development Workflow

Use [Getting Started](../README.md#getting-started) for the entry point,
[CONTRIBUTING](../CONTRIBUTING.md) for the contribution and pull-request workflow, and
[Building](Building.md#quick-start-guide) for setup and build procedures. This document adds only
the testing decision: which validation must accompany each change. Exact commands live in
[Developer Testing Commands and Authoring Reference](#developer-testing-commands-and-authoring-reference).

| Change | Minimum validation before push | GPU |
| --- | --- | --- |
| Backend, frontend, or SDK internal logic | Matching unit suite; then `hipdnn-unit-check` | Usually no |
| Public backend or frontend API | Black-box API/integration suite; then `hipdnn-integration-check` | Depends on path |
| Frontend-to-plugin routing or graph execution | End-to-end integration test | Depends on operation |
| Provider graph support or numerical behavior | Shared cross-provider bundle/test against reference executor | Yes |
| Provider-specific error, determinism, or tuning behavior | Provider integration suite | Yes |
| Bug fix | Regression test that fails before and passes after | Depends on defect |
| Build, package, install, or platform support | Relevant standalone/superbuild and install flow | Depends on scope |
| Performance-sensitive behavior | Manual before/after benchmark; no automated gate exists | Yes |

## Testing Strategy and Layers

### Unit Test Strategy

**Purpose:** validate one isolated piece of logic with external dependencies replaced by fakes or
mocks so a real provider and GPU are not required.

| Component | Location | Primary scope |
| --- | --- | --- |
| Backend | `backend/tests/` | Descriptors, plugin system, handles, graph extensions, errors, utilities |
| Frontend | `frontend/tests/` | Attributes, nodes, graph construction, validation, utilities |
| Data SDK | `data_sdk/tests/` | Data objects, logging, utilities |
| FlatBuffers SDK | `flatbuffers_sdk/tests/` | Serialized graph objects and helpers |
| Plugin SDK | `plugin_sdk/tests/` | Plugin API utilities and engine base classes |
| Test SDK | `test_sdk/tests/` | CPU references, validation helpers, test utilities |
| Provider | `dnn-providers/<name>/tests/` | Provider internals that do not require end-to-end graph execution |

GoogleTest is the test framework and GoogleMock is used for dependency seams. Unit tests must be fast,
isolated, deterministic, and runnable on Windows and Linux. Any test that truly needs a device must
use `SKIP_IF_NO_DEVICES()`; the skip means the configuration was not covered, not that GPU behavior
passed.

The `unit` CTest label selects the current unit binaries listed in
[`test_categories.yaml`](../test_categories.yaml). Naming and `TEST`/`TEST_P`/`TYPED_TEST` guidance live
in [Quick Reference](#quick-reference).

**Not covered by unit tests:** real provider discovery across an installed deployment, numerical
correctness of provider kernels, ASIC-specific behavior, and full frontend-to-provider execution.

### Integration Test Strategy

**Purpose:** validate behavior across public interfaces, component boundaries, real runtime layers,
or GPU hardware that an isolated unit test cannot represent.

| Layer | Location | What it proves | Real GPU/provider |
| --- | --- | --- | --- |
| Backend API | `tests/backend/` | Public C API behavior, descriptor lifecycle, execution and plugin-management contracts | Fake plugins; GPU only for GPU paths |
| Frontend-backend | `tests/frontend/` | Graph creation, descriptor conversion, routing, and execution flow | Fake plugins; no numerical solution validation |
| Provider integration | `dnn-providers/<name>/integration_tests/` | Provider-specific errors, determinism, support, and tuning behavior | Yes |
| Cross-provider | `dnn-providers/integration-tests/` | Same graph executed through every registered provider plugin and compared with a reference | Yes |

The shared suite builds one `hipdnn_integration_tests` binary:

- Each provider runs the binary against its own plugin. The same graph test can therefore cover
  MIOpen, hipBLASLt, and hip-kernel engines.
- Unsupported operations skip through a shared support check.
- Provider-owned TOML files define engine-specific tolerance overrides and skips.

See [Provider Integration](../../../dnn-providers/integration-tests/README.md#provider-integration)
for wiring and execution details.

Choose the test format based on what the test needs to prove:

- **Bundle:** Default for running a graph and comparing its output with a reference. Use one graph
  JSON for one case, or a template plus `sweep.json` for many shapes, data types, or layouts.
- **C++:** Use for behavior that graph data cannot express, such as error paths, API contracts,
  serialization, determinism, and benchmarking controls.

See [Two Ways to Test a Graph](../../../dnn-providers/integration-tests/README.md#two-ways-to-test-a-graph)
and [Bundle Formats](../../../dnn-providers/integration-tests/README.md#bundle-formats-single-graph-vs-template-sweep)
for authoring, migration, and golden-data details.

Put general graph correctness tests in the shared suite. Use a provider-specific integration
directory only for behavior unique to that provider.

Current limitations:

- Core `quick`, `standard`, `comprehensive`, and `full` categories run the same tests because `quick`
  matches all tests and the higher tiers add no patterns.
- The shared GPU-reference/framework category file enables `quick` and `standard`; its
  `comprehensive` and `full` blocks are disabled. MIOpen- and hipBLASLt-wrapped registrations for
  the external `hipdnn_integration_tests` binary do define all four tiers.
- Data-driven bundles require `--allow-bundles`; current provider CMake callers do not enable them,
  so the wider provider tier registrations still omit bundle cases.
- Exact TheRock package-test selection depends on live configuration outside this repository.

#### Existing test workarounds and tolerance overrides

The cross-provider harness supports provider-owned TOML files for numerical tolerance overrides and
test skips. Current files include:

- [MIOpen](../../../dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml): widened convolution and
  batch-normalization tolerances plus architecture-specific fusion skips.
- [hipBLASLt](../../../dnn-providers/hipblaslt-provider/config/HIPBLASLT_ENGINE.toml): a widened BF16
  matmul tolerance and a gfx12 FP16 bias-fusion skip.
- [hip-kernel MLOps](../../../dnn-providers/hip-kernel-provider/config/HIP_MLOPS_ENGINE.toml): flaky
  RMSNorm backward and slow large-batch LayerNorm skips.
- [hip-kernel ASM SDPA](../../../dnn-providers/hip-kernel-provider/config/ASM_SDPA_ENGINE.toml): wider
  backward-gradient tolerances.

These controls keep known numerical or support limitations visible and let unaffected coverage run.
They are not proof that the underlying behavior is correct. Not every entry has a public tracking
issue, owner, review date, or removal criterion, so stale or overly broad workarounds are a formal
gap below.

### Performance and Benchmarking

[`ROCm/dnn-benchmarking`](https://github.com/ROCm/dnn-benchmarking) uses the hipDNN Python bindings
to run graph workloads defined in JSON through installed engine plugins. It reports kernel time and
synchronized end-to-end time in a common JSON format. Optional PyTorch support can validate results
and compare ROCm with CUDA offline.

Current status:

- Runs are manual and have no set cadence.
- A weekly performance-regression run is planned.
- No automated baseline comparison, regression threshold, or gate exists yet.

The [Roadmap](Roadmap.md#benchmarking--performance-testing) tracks Windows support and CI/CD work.

| Item | Current state |
| --- | --- |
| Platform | [`ROCm/dnn-benchmarking`](https://github.com/ROCm/dnn-benchmarking) |
| Metrics available | HIP-event kernel timing and synchronized end-to-end timing |
| Workloads | hipDNN graph workloads defined in JSON and executed through the Python bindings |
| Result format | `SuiteResult` JSON; no component-owned baseline store or dashboard |
| Per-architecture baselines | Not established for automated comparison |
| Regression threshold | Not defined |
| PR-level gate | No |
| Scheduled cadence | Manual only today; weekly performance-regression run planned |
| Release qualification | No performance gate in the current Test Plan |

Performance-sensitive changes require a manual same-architecture before/after run with
`dnn-benchmarking` and reviewer-visible results. That manual process remains the current signal until
the planned weekly run, stable baselines, and ownership exist.

## Pre-submit and CI Gates

A workflow can run and fail without being configured as a required branch-protection check. Required
check settings are not stored in this repository. The table records checked-in behavior only.

Pre-submit is intentionally a subset of total validation. The hipDNN Superbuild PR gate is CPU-only;
its GPU-dependent tests skip. For operation/provider correctness, the trusted pre-submit signal is
the quick/smoke set from separate on-device CI; it does not include wider workload tiers, bundle
sweeps, performance testing, or sanitizer test execution. Core CTest currently labels all registered
core binaries as `quick`, but that registration boundary still omits the wider surfaces listed
above. A green pre-submit run therefore means the fast selected scope passed, not that full testing
completed.

| Validation | Trigger and configuration | Classification | Ownership |
| --- | --- | --- | --- |
| [hipDNN Superbuild CI](../../../.github/workflows/hipdnn-superbuild-ci.yml) | Applicable hipDNN/provider PRs; CPU-only Linux and Windows builds target gfx942/gfx1151 artifacts, run registered root CTest and frontend wheel tests, and skip GPU-dependent tests | Trusted CPU/build/quick signal; no on-device validation; required-check status not repository-visible | hipDNN + provider teams own tests; repository CI team owns workflow |
| [Primary TheRock CI](../../../.github/workflows/therock-ci.yml) | PRs and pushes; changed-project selection; wrapper defaults to `standard`, while provider pre-submit coverage remains the quick/smoke subset; docs/config-only paths and `skip-therockci` can skip | Trusted when it runs; exact package matrix comes from external live config | TheRock and CI infrastructure owners |
| [Pre-commit](../../../.github/workflows/pre-commit.yml) | PRs to `develop`; changed files only | Trusted formatting/lint signal | Repository CI owners |
| [Libraries PR Bot](../../../.github/workflows/libraries-pr-bot.yml) | PR policy checks | Trusted policy signal; source-change unit-test check is warning-only | Repository CI owners |
| Coverage | Local report generation only | Informational/manual; no PR upload or floor | hipDNN core |
| ASAN | Standalone check run manually; PR workflow is label-gated and build-only | Manual signal; no sanitizer test gate | hipDNN core |
| Performance | Manual `dnn-benchmarking` comparison | Manual signal; no PR/nightly automation | hipDNN core |
| Release verification | Manual Test Plan and recorded evidence | Release signal, not PR gate | Component team + release stakeholders |

### PR test classification and flaky tests

- **Trusted:** deterministic build, unit, API, integration, wheel, pre-commit, and policy checks that
  execute for the changed scope. Failures require investigation.
- **Informational/manual:** coverage reports, manual sanitizer runs, manual benchmarks, and release
  evidence not produced by a required PR workflow.
- **Unstable/flaky:** hipDNN has no encoded quarantine, owner, expiry, or tracking convention. Do not
  treat retry success as proof. File a tracking issue, preserve the original failure, assign an owner,
  and restore trusted status after fixing the cause.

PR workflows are not automatically retried. Nightly workflow failures may be retried on fresh
runners up to four total attempts for infrastructure failures; attempt history remains evidence and
must not be hidden.

## Coverage

hipDNN uses LLVM source-based coverage. Configure with `-DHIPDNN_ENABLE_COVERAGE=ON`; targets include
`unit-coverage`, `integration-coverage`, and `coverage`, with text, HTML, and LCOV output under
`coverage-report/`.

The measured objects are the backend library plus selected backend, frontend, public-backend, and SDK
test binaries. The report excludes dependencies, tests, generated data objects, `HipErrorHandler`,
and Test SDK mocks. It does not include provider objects, and `hipdnn_public_frontend_tests` is not
in the coverage object list. See [`CMakeLists.txt`](../CMakeLists.txt) for the authoritative target and
exclusion lists.

The current 80% overall and per-component figure is a **goal, not an automated acceptance threshold**.
No checked-in hipDNN workflow uploads coverage or enforces a floor, and no current measured baseline
is published here.

Keep two concepts separate:

- **Code coverage:** which instrumented lines or branches executed in the scoped report.
- **Test coverage:** which intended behaviors, error paths, operating systems, providers, GPU
  architectures, and deployment configurations were validated.

A high code-coverage percentage does not prove broad test coverage. New hardware-independent logic
should add a unit test; new public API behavior needs a black-box test; new provider behavior needs
real-device validation; bug fixes need a regression test.

## Nightly Validation

| Workflow | Cadence | Configuration | Limitation |
| --- | --- | --- | --- |
| [TheRock CI Nightly](../../../.github/workflows/therock-ci-nightly.yml) | Daily, 07:00 UTC | `comprehensive`; Linux gfx94X/gfx950, Windows gfx1151 | Exact hipDNN package tests depend on external live config |
| [TheRock Multi-Arch CI Nightly](../../../.github/workflows/therock-multi-arch-ci-nightly.yml) | Daily, 07:00 UTC | Linux gfx94X/gfx950, Windows gfx1151 | Overlaps primary nightly; component inventory remains externally configured |
| [TheRock ASAN Nightly](../../../.github/workflows/therock-multi-arch-ci-asan-nightly.yml) | Daily, 07:00 UTC | Linux full ASAN; gfx94X/gfx950/gfx125X | No Windows lane; exact hipDNN test inventory and current pass history are not recorded here |

No weekly hipDNN workflow is checked in. MIOpen- and hipBLASLt-wrapped external integration
registrations define `comprehensive` and `full`, while both hip-kernel external registrations expose
only `standard`. Checked-in workflows do not establish a weekly full run, and bundle cases remain
disabled at the caller. The shared GPU-reference/framework category file has its own
`comprehensive` and `full` blocks disabled.

## Supported and Validated Configurations

This matrix describes observed automated validation, not the complete product support matrix. A build
target is not evidence that tests executed on that target.

| Configuration | Automated validation | Frequency | Notes |
| --- | --- | --- | --- |
| Superbuild PR, Linux target gfx942 | CPU-only build, registered core/fake-plugin CTest, and frontend wheel tests | PR | `gfx942` selects build artifacts/targets; no GPU is present and GPU tests skip |
| Superbuild PR, Windows target gfx1151 | CPU-only build, registered core/fake-plugin CTest, and frontend wheel tests | PR | `gfx1151` selects build artifacts/targets; no GPU is present, GPU tests skip, and clang-tidy is disabled |
| TheRock Linux gfx94X | On-device package tests when selected by the live package matrix | PR/push/config-dependent | Exact hipDNN inventory is externally configured |
| TheRock Windows gfx1151 | On-device package tests when selected by the live package matrix | PR/push/config-dependent | Exact hipDNN inventory is externally configured |
| Linux gfx950 | Built by primary TheRock matrix but package tests are skipped there; nightly configured | Nightly/config-dependent | Do not count PR build as test coverage |
| Linux gfx125X | Built by primary TheRock matrix but package tests are skipped there; full-ASAN nightly configured | Nightly/config-dependent | Do not count PR build as test coverage |
| Other GPU families | Provider-owned where supported | Provider/release dependent | No hipDNN-owned automated matrix is documented |
| CPU-only | Core unit and fake-plugin suites can run; GPU tests skip | Local/CI-dependent | A skip records no GPU validation |

Per-ASIC numerical correctness is delegated to providers and the cross-provider suite. hipDNN owns
routing and API correctness. Both must be present before claiming end-to-end coverage on an ASIC.

## ASAN, TSAN, and Sanitizer Coverage

| Sanitizer | Current coverage | Cadence/gating | Explicitly not covered |
| --- | --- | --- | --- |
| Standalone ASAN, Linux | `BUILD_ADDRESS_SANITIZER=ON`; requires ASAN-enabled ROCm; recommended `ctest -L standard` | Manual; current trusted signal | Availability depends on suitable ROCm build |
| Standalone ASAN, Windows | Instruments hipDNN-built code, not installed ROCm libraries | Manual; current trusted signal | No dated clean baseline or tracked failure list is checked in |
| TheRock HOST_ASAN/full ASAN on PR | Workflow is label-gated; TheRock currently disables sanitizer test runners for PR events | Opt-in build-only; not a signal | Runtime defects are not exercised on PRs |
| TheRock full ASAN, Linux | Scheduled workflow requests device-side ASAN for gfx94X/gfx950/gfx125X | Scheduled infrastructure; not yet an owned hipDNN signal | Exact hipDNN inventory, pass history, and failure ownership are not recorded here |
| TSAN | Linux-only CMake support exists | Manual option only; no documented successful run or CI | No established baseline for data races |
| UBSAN/MSAN | No hipDNN configuration or workflow found | None | Undefined behavior and uninitialized-read sanitizer coverage |

For current testing-strategy reporting, ASAN is manually run. The scheduled Linux workflow is useful
infrastructure, but it does not replace manual evidence until hipDNN's executed inventory, pass
history, and failure ownership are established.

Manual ASAN commands are documented in
[Building: Address Sanitizer Build](Building.md#address-sanitizer-build). ASAN and TSAN are
mutually exclusive.

Known exclusions are part of the result, not passing evidence:

- `hipdnn_backend_logging_shutdown_tests` is not registered under standalone ASAN, TheRock ASAN, or
  HOST_ASAN because the test intentionally performs an unclean threaded shutdown.
- Standalone `BUILD_ADDRESS_SANITIZER` disables four sample tests on Windows and seven on Linux; the
  three additional Linux exclusions are fused-convolution tests. Comments cite provider/driver hangs
  but no tracking issue or re-enable criterion.
- `SKIP_IF_ASAN()` and `SKIP_IF_TSAN()` exist in Test SDK, but no current call sites were found.
- A missing `llvm-symbolizer` warns rather than failing configuration, so reports may be unsymbolized.

Upstream [TheRock issue #3433](https://github.com/ROCm/TheRock/issues/3433) tracks sanitizer test-runner work. hipDNN has no component-owned sanitizer tracking issue referenced in this repository.

## Static Analysis

clang-tidy is hipDNN's current static analyzer. The complete `hipdnn-tidy` target is run manually and
is limited to hipDNN C++ sources, headers, and tests represented in hipDNN's compilation database.
Provider components and HIP-language code are outside that scan because clang-tidy does not reliably
analyze HIP compilation.

Builds can opt eligible C++ targets into per-target clang-tidy checks, but hipDNN does not have a
dedicated full static-analysis job on a regular cadence. The current owned signal is therefore the
manual `hipdnn-tidy` run, not component-wide or scheduled analysis.

## For New Contributors

### Choosing the Right Test Type

1. Can the behavior be validated without a real provider or GPU?
   - Yes: add a unit test next to the component.
   - No: add an integration test at the narrowest boundary that observes it.
2. Is it public API behavior?
   - Add or update a black-box API test.
3. Is it graph numerical correctness on a provider?
   - Add a cross-provider data-driven graph test and compare against the reference executor.
4. Is it provider-specific behavior beyond graph correctness?
   - Add a provider integration test.
5. Is it a bug fix?
   - Add a regression test proven to fail before the fix.
6. Is it performance-sensitive?
   - Record a manual before/after benchmark; automated performance protection does not yet exist.
7. Does it add an OS or GPU configuration?
   - Run on that configuration and update the matrix and gap ledger here.

| Change type | Expected validation |
| --- | --- |
| Hardware-independent logic | Unit test |
| On-device routing behavior | End-to-end hipDNN integration test |
| Provider numerical behavior | Cross-provider reference comparison |
| Public API | Black-box API test |
| Bug fix | Red/green regression test |
| Performance-sensitive path | Before/after benchmark and reviewer-visible results |
| New GPU or OS support | Validation on that configuration plus matrix update |
| Packaging/build change | Superbuild, install, and artifact test as applicable |

## Key Quality Concerns

| Concern | Why it matters | Primary validation |
| --- | --- | --- |
| Frontend-to-plugin data path | hipDNN must pass tensors, attributes, and scalars faithfully | Frontend/backend integration and real-device routing tests |
| Numerical correctness | Executed graph outputs must match expected results | Cross-provider suite against CPU reference |
| Public API compatibility | Backend C and frontend C++ APIs are consumer contracts | Black-box backend API and frontend suites |
| Cross-platform behavior | Windows and Linux are supported but differ in toolchain and GPU coverage | Superbuild CI on both OSes plus explicit configuration matrix |
| Memory safety | Plugin loading, descriptor ownership, and failure paths must not leak or corrupt memory | Manually run ASAN; automated signal gap remains |
| Performance | Routing and provider changes must not regress throughput or latency | Manual `dnn-benchmarking` comparison; automated signal gap remains |

## Release Validation

Release and milestone verification is manual and evidence-based. Follow the
[Release and Milestone Test Plan](#release-and-milestone-test-plan), record exact artifacts, commits, hardware, commands, and output
in the [Test Run Record Template](#test-run-record-template), and review this document's configuration
and gap tables before sign-off.

A release record must distinguish:

- build success from tests executed;
- skipped tests from passing tests;
- hipDNN routing coverage from provider numerical/ASIC coverage;
- post-merge/nightly evidence from final release-artifact evidence;
- accepted known gaps from accidentally untested scope.

A PR merged to `develop` is not release validation. Sign-off requires evidence tied to the artifact
being released and an explicit review of unresolved high-risk gaps.

## Dependencies and Validation Handoffs

| Dependency/area | Owner | Validation responsibility | Known handoff risk |
| --- | --- | --- | --- |
| hipDNN core | `@ROCm/hipdnn-core` | Component unit, API, routing, documentation |
| Providers and shared integration | `@ROCm/dnn-providers-core` | Numerical correctness and supported ASIC coverage | Provider support matrix and CI inventory are not centralized here |
| Repository CI | `@ROCm/rocm-libraries-reviewers` | Workflow definitions, runners, shared policy | Branch required-check settings are external to repository files |
| TheRock / `therock-ci-config` | TheRock/CI owners | Build, package, and live test matrices | Exact package selection can change outside this repository |
| QA/release stakeholders | Not recorded here | Release qualification and sign-off | Ownership and exception process are not documented |
| Downstream consumers | Framework teams | Compatibility/system validation | No component-owned pre-merge downstream signal is documented |

## Improvement Roadmap

### Near term

1. Publish the actual branch-protection required-check list and keep it synchronized with this file.
2. Enable sanitizer test runners for labeled PR workflows and decide whether one sanitizer lane should
   become required; coordinate with TheRock issue #3433.
3. Create hipDNN tracking issues for Windows ASAN failures and every sanitizer exclusion, with owner,
   affected configurations, and re-enable criteria.
4. Publish a repeatable coverage baseline for the current scoped object list and resolve whether 80%
   remains an aspiration or becomes an enforced floor.
5. Assign a named technical lead/document steward and validation-area owners.

### Medium term

1. Wire cross-provider data bundles into registered provider CI.
2. Define real incremental content for core `standard`, `comprehensive`, and `full` tiers; give the
   MIOpen/hipBLASLt wider tiers an explicit owned cadence, decide the missing hip-kernel tiers, and
   enable bundle execution.
3. Establish one reproducible TSAN run, baseline its failures, and decide whether scheduled CI is
   warranted.
4. Make sanitizer failure semantics, leak detection, symbolization, and exclusions explicit and
   consistent across standalone, full ASAN, and HOST_ASAN modes.
5. Publish exact nightly hipDNN inventory and failure ownership.

### Longer term

1. Add stable per-architecture performance baselines, a small PR-visible benchmark signal, and a
   broader nightly suite before considering a blocking threshold.
2. Assess host-side UBSAN and, only with instrumented dependency support, MSAN.
3. Add downstream compatibility coverage where escaped regressions show it is needed.

## Known Risks and Gaps

An empty or `Untracked` tracking cell means the gap is acknowledged but has no component-owned work
item referenced here. Per-gap assignees belong on work items, where ownership can change without
making this document stale.

### Sanitizers and memory safety

| Gap | Regression risk | Impact | Mitigation today | Tracking |
| --- | --- | --- | --- | --- |
| ASAN signal is manually run today; PR sanitizer jobs are build-only and scheduled hipDNN inventory/ownership is not established | High | High | Manual `ctest -L standard` | [TheRock #3433](https://github.com/ROCm/TheRock/issues/3433); hipDNN item untracked |
| Windows has no ASAN CI and no dated clean baseline or tracked failure inventory | High | High | Manual Windows ASAN build/run | Untracked |
| Sanitizer exclusions lack issue IDs, owners, expiry, and consistent mode handling | Medium | High | Exclusions are visible in CMake | Untracked |
| TSAN has build support but no documented successful run or CI | Medium | High | Standard tests and code review | Untracked |
| UBSAN and MSAN are absent | Medium | Medium | Compiler warnings and manual hipDNN-only clang-tidy | Untracked |
| Missing symbolizer only warns; fail-on-error/leak settings are not an explicit cross-platform policy | Medium | Medium | Runtime defaults and CTest failure handling | Untracked |

### Coverage and test surface

| Gap | Regression risk | Impact | Mitigation today | Tracking |
| --- | --- | --- | --- | --- |
| 80% coverage is not enforced and no current baseline is published | Medium | Medium | Manual LLVM coverage targets | Untracked |
| Coverage scope omits providers and `hipdnn_public_frontend_tests` | Medium | Medium | Integration tests still execute outside report | Untracked |
| Core `quick`, `standard`, `comprehensive`, and `full` tiers are identical | Medium | Medium | Full current core suite still runs under every tier | Untracked |
| Cross-provider bundles are not wired into registered provider CI | High | High | Manual/migration-pipeline bundle runs | Untracked |
| Cross-provider tier definitions differ: MIOpen/hipBLASLt register `comprehensive`/`full`, hip-kernel exposes only `standard`, cadence is not established in checked-in workflows, bundles remain disabled, and no weekly lane exists | Medium | High | Existing provider category labels and manual bundle runs | Untracked |
| No encoded flaky-test ownership, expiry, or quarantine process | Medium | Medium | Human triage; nightly infrastructure retry history | Untracked |
| Tolerance overrides, architecture skips, flaky-test skips, and runtime workarounds have no single tracking/owner/expiry policy; some entries have no public issue or removal criterion | High | High | Provider TOML keeps controls and reasons visible; code review | Untracked |

### CI, configurations, and ownership

| Gap | Regression risk | Impact | Mitigation today | Tracking |
| --- | --- | --- | --- | --- |
| Required branch-protection checks are not visible in repository files | Medium | Medium | Workflow definitions and PR status UI | Untracked |
| Pre-submit validates only the quick/smoke subset, excluding wider workload tiers, bundle sweeps, performance, and sanitizer execution | High | High | Wider manual and post-merge validation | Untracked |
| Exact TheRock hipDNN package-test matrix comes from live external configuration | Medium | Medium | This dated document and workflow logs | Untracked |
| Linux gfx950/gfx125X are built but package tests are skipped in primary TheRock PR CI | Medium | High | Nightly and provider-owned validation where configured | Untracked |
| Windows validates a different subset and disables clang-tidy in superbuild CI | Medium | Medium | Manual hipDNN-only clang-tidy plus Windows CPU/build tests | Untracked |
| Static analysis is manual, limited to hipDNN C++ code, excludes provider components and HIP-language code, and has no regular cadence | High | High | Manual `hipdnn-tidy` on eligible hipDNN sources | JIRA ID: ALMIOPEN-2300 |
| No named technical lead, validation-area owners, or release exception owner is recorded | Medium | Medium | CODEOWNERS teams | Untracked |
| Mainline/NPI and pre-silicon validation expectations are not documented | Medium | Medium | Release-specific coordination | Untracked |

### Performance and downstream validation

| Gap | Regression risk | Impact | Mitigation today | Tracking |
| --- | --- | --- | --- | --- |
| `dnn-benchmarking` is the performance platform, but no automated PR or nightly regression signal consumes it | High | High | Manual same-architecture before/after runs | JIRA ID: ALMIOPEN-1908 |
| No owned per-architecture baseline store, comparison threshold, or failure-routing policy | High | High | `dnn-benchmarking` emits comparable JSON results | JIRA ID: ALMIOPEN-1908 |
| No documented downstream compatibility gate before merge | Medium | High | Framework/system testing after integration | Untracked |

## Owners and Review Cadence

The hipDNN CODEOWNERS team owns this document until a named steward and technical lead are recorded.
CI workflow ownership, provider test ownership, and release ownership remain split as shown in
[Dependencies and Validation Handoffs](#dependencies-and-validation-handoffs).

Review this document:

- at least quarterly, with explicit attention to whether the gap tables are shrinking;
- whenever a workflow changes what it runs or what it gates;
- when a new test layer, provider, platform, GPU family, sanitizer, or performance signal is added;
- after a significant escaped regression;
- before a major release, together with supported configurations and accepted gaps;
- when NPI or release-validation assumptions change.

A gap is not complete because related code merged. Close a row only after the intended validation is
running at the documented cadence, has an owner, and has evidence that the gap no longer exists.

---

## Detailed Testing Layer Reference

This document describes hipDNN's unit, API, and integration test layers. The authoritative current-state strategy, including CI signals, supported configurations, sanitizer coverage, ownership, roadmap, and known gaps, is [this document](#hipdnn-testing-strategy).

Refer to [Coding Style and Naming Guidelines](CodingStyleAndNamingGuidelines.md) for test naming conventions.

---

### 1. White Box Testing (Unit Tests)

White box tests focus on internal implementation details of hipDNN components.

#### Component Comparison

| Component | Location | Purpose | GPU Testing | Environments |
|-----------|----------|---------|-------------|--------------|
| **Backend** | `backend/tests/` | Test internal implementation of hipDNN backend | Minimal/None | Windows & Linux |
| **Frontend** | `frontend/tests/` | Test internal implementation of hipDNN frontend | Minimal/None | Windows & Linux |
| **Data SDK** | `data_sdk/tests/` | Test internal implementation of Data SDK | Minimal/None | Windows & Linux |
| **Plugin SDK** | `plugin_sdk/tests/` | Test internal implementation of Plugin SDK | Minimal/None | Windows & Linux |
| **Test SDK** | `test_sdk/tests/` | Test internal implementation of Test SDK | Minimal/None | Windows & Linux |
| **Provider** | `dnn-providers/<name>/tests/` | Test internal implementation of a specific provider | Minimal & fast | Windows & Linux |

Note: If a test depends on the GPU then it needs to be marked with `SKIP_IF_NO_DEVICES()` so tests run and pass correctly on CPU only machines.
---

#### Test Categories by Component

##### Backend
- Descriptors
- Plugin system
- Error handling
- Backend utilities
- Handle
- Graph extensions

##### Frontend
- Attribute
- Node
- Graph construction & flow
- Frontend utilities

##### Data SDK
- Data objects
- Logging
- SDK utilities

##### Plugin SDK
- Plugin API utilities
- Engine base classes

##### Test SDK
- CPU reference implementations
- Test utilities
- Validation helpers

##### Plugin
- TBD based on plugin implementation
- See the recommended implementation in [Plugin Development](PluginDevelopment.md#implementation-details)

#### Common Requirements
- **Mocking**: Use GMOCK for mocking dependencies
- **Execution**: Fast execution required (Time limits enabled in TheRock CI)
- **Isolation**: Use stubbed/mocked implementations for dependencies
- **GPU Operations**: Must be marked with `SKIP_IF_NO_DEVICES()`
- **Coverage**: Each component should maintain >80% code coverage

---

### 2. Integration Tests

#### Black Box Integration Tests

Black box tests validate the public API without knowledge of internal implementation.  These are a type of integration test.

##### Backend API Tests

| Attribute | Details |
|-----------|---------|
| **Location** | `tests/backend/` |
| **Purpose** | Validate API of hipDNN backend works as expected |
| **Requirements** | • Test only public interfaces from `backend/include/`<br>• Use stubbed plugins for controlled testing<br>• Fast running<br>• GPU operations marked with `SKIP_IF_NO_DEVICES()` |
| **Environments** | Windows & supported Linux distros |
| **Frequency** | Run on each PR |

###### Test Categories
- Descriptor APIs (create, get/set properties, destroy)
  - Engine API
  - Engine config API
  - Engine heuristic API
  - Execution plan API
  - Handle API
  - Variant pack API
  - Graph API
  - Graph extension API for serialized graph structures
- Backend execute API
- Plugin management extension API
- Logging extension API

---

#### End to End Integration Tests

Integration tests validate end-to-end functionality across components.

##### End to End Integration Test Comparison

| Test Type | Location | Purpose | GPU Required | Test Speed | Environments |
|-----------|----------|---------|--------------|------------|--------------|
| **Frontend-Backend** | `tests/frontend/` | Validate end-to-end hipDNN functionality | No - mark GPU ops with `SKIP_IF_NO_DEVICES()` | Fast | Windows & Linux |
| **Provider Integration** | `dnn-providers/<name>/integration_tests/` | Validate end-to-end graph support for a provider | Yes - required for validation | Can be slower | Windows & Linux |
| **External Integration** | `dnn-providers/integration-tests/` | Run graphs through a hipDNN provider plugin and compare results against a reference executor, across providers | Yes - required for validation | Can be slower | Windows & Linux |

##### Test Requirements by Type

###### Frontend-Backend
- Use fake plugins for controlled behavior
- No accuracy/solution validation (stubbed)
- Test graph creation and execution API
- Test backend descriptor creation from frontend
- Test execution flow validation

###### Provider Integration
- Validate correctness and graph support
- Each provider maintains its own test suite
- Test on all ASICs supported by the provider
- Tests are divided into two categories described by the prefix argument passed to INSTANTIATE_TEST_SUITE_P
  - **Smoke** - These tests are designed to test features using the smallest possible shape and run quickly (combined smoke test run time must be under 5 mins)
  - **Full** - These tests can contain regression shapes, large shapes, or slow shapes

###### External Integration
- Each provider runs the same `hipdnn_integration_tests` binary against its own plugin.
- Superbuilds discover the plugin automatically. For standalone runs, pass
  `--test-article /path/to/plugin.so`.
- Tests use four tiers: `Smoke`, `Standard`, `Comprehensive`, and `Full`. Higher tiers include lower
  tiers. Tests without a tier prefix run in smoke.
- Use the [cross-provider guide](../../../dnn-providers/integration-tests/README.md#two-ways-to-test-a-graph)
  to choose between bundles and C++ tests. It also documents
  [single-graph and template-sweep bundles](../../../dnn-providers/integration-tests/README.md#bundle-formats-single-graph-vs-template-sweep).
- Provider-owned TOML files can override tolerances or skip cases for known limitations. These are
  workarounds, not proof that skipped behavior works. See
  [Known Risks and Gaps](#known-risks-and-gaps).

#### Graph Validation
Outputs can be checked against precomputed golden tensors, a GPU reference, or a CPU reference. See
the [verification modes](../../../dnn-providers/integration-tests/README.md#verification-modes) for
selection rules and the [CPU Graph Executor Design Document](rfcs/0001_CpuGraphExecutorDesign.md)
for CPU implementation details.

---

### Validation Responsibility and GPU Coverage

hipDNN routes graphs to provider plugins; providers execute them. Testing follows that split:

- **hipDNN** owns routing correctness. Its backend and end-to-end tests check that tensors,
  attributes, and results move through the plugin boundary correctly. hipDNN does not implement
  kernels, so it does not own a per-`gfx` numerical matrix.
- **Providers** own numerical correctness on supported GPUs. Each provider runs the shared suite,
  which executes graphs through its plugin and compares results with a reference.

What hipDNN itself validates directly, by OS:

| Configuration | Validated | Notes |
|---------------|-----------|-------|
| Linux | Yes | Registered core and provider suites; data-driven bundles are currently omitted |
| Windows | Yes | Registered subset; platform exclusions apply and data-driven bundles are omitted |
| Specific `gfx` targets | Delegated to providers | On-device tests exercise whatever GPU is present; per-ASIC operation correctness is each provider's responsibility |

> [!NOTE]
> Cadence differs by suite and workflow. Core `quick`, `standard`, `comprehensive`, and `full` currently select the same tests. The shared GPU-reference/framework category file disables `comprehensive` and `full`; MIOpen- and hipBLASLt-wrapped external registrations define them, while hip-kernel external registrations expose only `standard`. Bundle cases remain disabled at the caller. See [TESTING.md § Pre-submit and CI Gates](#pre-submit-and-ci-gates) for verified current behavior.

### Key Quality Concerns

The following are the highest-priority correctness guarantees, and how each is validated:

- **Frontend-to-plugin data path** - hipDNN must pass the correct tensors, attributes, and scalars to the selected provider and return the plugin's results faithfully. Validated by the backend/end-to-end integration tests on a real device.
- **Numerical correctness of operations** - the results of executed graphs must be correct. Validated by the cross-provider integration suite, which runs graphs through a provider plugin and compares the output against a reference executor.
- **Public API compatibility** - the backend C API and frontend C++ API are the contract for consumers. Validated by the black-box API tests (`tests/backend/`) and frontend tests.
- **Cross-platform (Windows and Linux)** - the library must build and behave correctly on both. Validated by running the suites on both OSes, with GPU-dependent tests guarded by `SKIP_IF_NO_DEVICES()`.
- **Memory safety** - no leaks or invalid accesses. The current trusted sanitizer signal is a manually run standalone check; see [ASAN, TSAN, and Sanitizer Coverage](#asan-tsan-and-sanitizer-coverage).

---

### 3. General Testing Requirements

#### Code Coverage
- **Tool**: LLVM source-based coverage (`-fprofile-instr-generate -fcoverage-mapping`, then `llvm-profdata` + `llvm-cov`). Enable with `-DHIPDNN_ENABLE_COVERAGE=ON` and build the `coverage` target; reports land in `coverage-report/` and include LCOV output.
- **Target**: 80% overall and per component is the current goal.
- **Enforcement**: No checked-in hipDNN workflow uploads coverage or enforces a threshold. Current measured baselines are not published.
- **Scope**: The instrumented object list excludes providers and `hipdnn_public_frontend_tests`; see [TESTING.md § Coverage](#coverage).

> [!NOTE]
> Code coverage measures execution of the instrumented object set. It does not establish behavior, provider, OS, or GPU-configuration coverage.

#### Test Environment Compatibility

Tests must work in the following environments:

| Environment Type | Supported Methods |
|-----------------|-------------------|
| **CLI Build Environment** | `ninja hipdnn-check`, `ninja hipdnn-check-verbose` (standalone also has the unprefixed `check`) |
| **IDE** | Visual Studio Code and extensions like TestMate |
| **Artifacts** | • Installed testing artifacts<br>• Running built test executables |
| **Operating System** | • Windows<br>• Supported Linux distros |

> [!TIP]
> `ninja hipdnn-unit-check` runs fast, isolated unit and API tests (also: `hipdnn-unit-check-verbose`).<br>
> `ninja hipdnn-integration-check` runs slower, end-to-end integration tests (also: `hipdnn-integration-check-verbose`).<br>
> In a standalone `projects/hipdnn` build the unprefixed aliases (`check`, `unit-check`, `integration-check`, ...) also exist. See [ctest vs. check targets](#ctest-vs-check-targets) for the full ctest/check-target mapping.

#### GPU Requirements
- **Without GPU**: All GPU tests must be skippable (warnings, not errors)
- **With GPU**: Tests should detect and utilize available GPU resources
- **Platform Support**: Windows & supported Linux distributions

#### Applicable CI

Inspect every workflow triggered for a pull request. The hipDNN Superbuild PR gate runs without a GPU, so GPU-dependent tests skip; architecture values select build artifacts rather than on-device validation. Path and label filters can skip other workflows, and checked-in files do not expose branch-protection required-check settings. See [TESTING.md § Pre-submit and CI Gates](#pre-submit-and-ci-gates).

#### Static Analysis

clang-tidy is limited to eligible hipDNN C++ sources, headers, and tests. The complete `hipdnn-tidy` scan is manual, excludes provider components and HIP-language code because clang-tidy does not reliably analyze HIP compilation, and has no regular cadence. See [TESTING.md § Static Analysis](#static-analysis).

#### Flaky Tests

Flaky tests are not an accepted final state. hipDNN has no encoded quarantine, owner, or expiry convention today; record the original failure, file a tracking issue, assign an owner, and restore trusted status after fixing the cause.

### 4. Performance Testing

[`ROCm/dnn-benchmarking`](https://github.com/ROCm/dnn-benchmarking) uses the hipDNN Python bindings
to run JSON graph workloads. Runs are manual today; weekly regression runs are planned. See
[Performance and Benchmarking](#performance-and-benchmarking) for current details.

---

## Release and Milestone Test Plan

This section is the **hipDNN milestone / release verification** plan: the procedures and expectations for validating a release build and confirming it is ready to ship. It is **not** the per-PR development workflow; see [Developer Testing Commands and Authoring Reference](#developer-testing-commands-and-authoring-reference) for local commands and the strategy above for current CI signals, ownership, configurations, and known gaps.

> [!IMPORTANT]
> **All prerequisites and tests in this section must pass for a successful release.**

---

### Objective

The objective of a verification run is to produce **defensible evidence** that a specific hipDNN build is correct and ready to ship. Executing this plan should answer three questions without ambiguity:

- **What was validated?** The exact build (OS, GPU family, ROCm version, source commit) that was exercised.
- **What was checked?** The prerequisites, test suites, and behaviors that were confirmed to pass.
- **Can it be trusted?** Whether an independent reader can reproduce the same result from the recorded steps.

#### Why the details must be recorded

A verification run only has value if its outcome can be tied back to a specific build and reproduced later. Record the identifiers and evidence as you go, because:

- **Traceability** - a passing result is meaningless unless it is pinned to the exact artifact and source commit it came from. Recording the build identifier and source SHA proves the tested code is the code being shipped.
- **Reproducibility** - the exact commands and observed output let someone re-run the validation and get the same result, rather than trusting a summary.
- **Auditability** - a release sign-off may be revisited weeks later (for a regression, a hotfix, or a compliance check). The record is the durable proof of what was done.

Capture the run using the [Test Run Record Template](#test-run-record-template), which provides the structure for the identifiers, evidence, and reproducible commands described below.

---

### Prerequisites

#### Test Case 1: Applicable CI Evidence Is Green

Record the exact workflows that ran for the source and artifact under qualification. Current checked-in automation includes:

| CI check | Verified scope |
|----------|----------------|
| hipDNN Superbuild CI | Pull requests only; CPU-only Linux/Windows build and test gate targeting gfx942/gfx1151 artifacts; GPU-dependent tests skip |
| Primary TheRock CI | Pull requests and pushes; selects changed projects and defaults to the `standard` tier; exact package tests depend on external live configuration |
| pre-commit | Pull requests to `develop` and selected pushes; checks changed files |
| TheRock nightlies | Post-merge comprehensive and multi-architecture workflows; exact hipDNN inventory depends on external live configuration |

Workflow definitions do not expose branch-protection required-check settings. Coverage is not a checked-in hipDNN CI gate. See [TESTING.md § Pre-submit and CI Gates](#pre-submit-and-ci-gates).

#### Test Case 2: Documentation is Current

Verify that all documentation is up to date:

1. Check version numbers throughout the documentation
2. Review instructions, explanations, and wording for clarity and accuracy
4. Verify changelog is complete and correct

> See the documentation listed in the [README](../README.md#documentation) to identify relevant areas.

---

### Running Tests From TheRock Builds

The hipDNN library is included in ROCm development and release builds produced by [TheRock](https://github.com/ROCm/TheRock). These builds ship the hipDNN test executables, so you can validate a release without building from source. The same download also provides a complete ROCm tree, which the [source build](#running-tests-from-source-build) section below reuses as its build dependency, so a single download serves both flows.

#### Obtain a ROCm Build with the hipDNN Tests

The hipDNN test executables and samples are not in the plain distribution tarball; they ship in the matching `-tests.tar.gz` variant. That variant is a superset of the plain distribution tarball (it contains the full ROCm tree plus the tests and samples), so download only the `-tests.tar.gz` and extract it into a `rocm-artifacts` folder. See [Tarballs](Building.md#tarballs) for the filename structure and how to pick a version.

```bash
mkdir rocm-artifacts

# Replace <platform>, <group>, and <version> to match the build under test.
curl -O https://rocm.nightlies.amd.com/tarball-multi-arch/therock-dist-<platform>-<group>-tests-<version>.tar.gz

tar -C rocm-artifacts -zxf therock-dist-<platform>-<group>-tests-<version>.tar.gz
```

The commands below assume the tarball was extracted to `./rocm-artifacts`; adjust the paths if you used a different folder.

**Record what was validated.** The tarball filename encodes the OS, GPU family, and ROCm version, but the authoritative record is the manifest inside the tree (`rocm-artifacts/share/therock/therock_manifest.json`), which also carries the exact source commit the build came from. Capture the ROCm version and the rocm-libraries source commit (`pin_sha`) in your test record; the [Test Run Template](#4-replication-setup) gives the exact commands and confirms the delivery commit is contained in that commit's history.

#### Running the hipDNN Tests

Use ctest to list the hipDNN test executables:
```
ctest --test-dir rocm-artifacts/bin/hipdnn --show-only
```
The installed inventory changes as suites are added. At the time of writing, CMake registers 11 installed CTest executables: the Backend, Frontend, Data SDK, FlatBuffers SDK, Plugin SDK, Test SDK, public Backend, public Frontend, backend-logging shutdown, and two frontend dynamic-load suites. Treat `ctest --show-only` from the artifact under test as authoritative and paste that observed inventory into the test record.

Run all hipDNN tests in parallel:
```
ctest --test-dir rocm-artifacts/bin/hipdnn --output-on-failure --parallel 8 --timeout 30
```
Record the observed output. A successful run ends with:

```text
100% tests passed, 0 tests failed out of <observed count>
```

Do not replace `<observed count>` with a copied sample value; the artifact's `ctest --show-only` inventory and run summary are the evidence.

Use the --verbose option for more detailed output:
```
ctest --test-dir rocm-artifacts/bin/hipdnn --parallel 8 --timeout 60 --verbose
```

---

### Running Tests From Source Build

This section builds hipDNN from source and runs the tests it produces. It can reuse the ROCm tree from the [TheRock build](#running-tests-from-therock-builds) above, so a single download serves both flows.

#### Obtain ROCm (Build Dependency)

Building from source needs ROCm installed as a **build dependency** (the compiler, HIP, and libraries).

- **Reuse the TheRock build (recommended).** If you extracted a `-tests.tar.gz` above, the `rocm-artifacts` folder is already a complete ROCm tree. Add its `bin` folder to your `PATH` so the standalone presets discover it automatically:
  ```bash
  export PATH="$(pwd)/rocm-artifacts/bin:$PATH"   # Linux; on Windows add rocm-artifacts\bin to PATH
  ```
- **Or install ROCm separately.** If you did not obtain a build above, install ROCm by any method in [Obtaining ROCm](Building.md#obtaining-rocm). You do **not** need the `-tests.tar.gz` variant here, because the tests are compiled from source.

Record the ROCm version you built against in your test record (for example the output of `hipconfig --version`), so the validated build can be identified later.

#### Test Case 1: Build and Run the Automated Tests

Build the standalone tests following the [Quick Start Guide](Building.md#quick-start-guide); for the superbuild, see [Superbuild](Building.md#superbuild). Then run them with ctest (the same command on Linux and Windows):

```bash
ctest --test-dir build/release
```

> On Windows, ensure the ROCm `bin` folder is on your `PATH` before running so the test executables can load the ROCm DLLs.

##### Expected Results

- **Test Status**: All tests should pass
- **GPU Test Behavior**:
  - **Without GPU**: All GPU tests should skip gracefully without failures
  - **With GPU**: hipDNN provider plugin integration tests may skip if the GPU is not supported
    - Skipped tests should provide clear messages indicating lack of ASIC support
- **Provider Support**: ASIC-specific coverage is determined by individual providers and is not a global hipDNN requirement

---

### ASAN Enabled Tests

#### Test Case 1: Build and Run the Automated Tests with ASAN Enabled

> [!NOTE]
> Standalone ASAN is manual on Linux and Windows. Linux also has an opt-in pull-request workflow that currently builds sanitizer variants without running their tests, plus a scheduled full-ASAN nightly. Windows has no ASAN CI lane. See [TESTING.md § ASAN, TSAN, and Sanitizer Coverage](#asan-tsan-and-sanitizer-coverage).

Build with address sanitizer enabled following the [Address Sanitizer Build](Building.md#address-sanitizer-build) instructions, then run the `standard` tier (`ctest --test-dir <build> -L standard`).

##### Expected Results

- **Test Status**: All registered tests must pass; every disabled or unregistered test must be recorded as an exclusion.
- **Memory Safety**: No sanitizer finding is acceptable in the executed scope.
- **Platform Evidence**: Record the exact OS, GPU, ROCm build, sanitizer mode, exclusions, and observed output. Do not claim a clean Windows baseline without a dated clean run.

---

## Test Run Record Template

This section is the recording artifact for a **hipDNN milestone / release verification** run. It captures the traceability, evidence, and reproducible commands that show a release build was validated and is ready to ship. It is **not** the per-PR development testing form; see [Developer Testing Commands and Authoring Reference](#developer-testing-commands-and-authoring-reference) for local commands and the strategy above for current status and gaps.

**How to use this template:**

- Copy this section to a working notes file for your feature or milestone (e.g. `ROCM-<ticket> <short name> validation.md`) and fill in every `<placeholder>`.
- The run this template records follows the [Release and Milestone Test Plan](#release-and-milestone-test-plan). Follow the plan's procedures and record results here.
- Replace the illustrative snippets with your own observed commands and output. Keep it evidence-based: paste real command output, not a summary.
- See [Release Validation](#release-validation) for how this template and the plan fit into the overall strategy.

**Filling this in as you follow the Test Plan.** The sections below are ordered to match the qualifier's path through the plan. Work top to bottom:

1. Record identifiers as you go (section 1) and describe what the run covers (section 2).
2. [Test Plan → Prerequisites](#prerequisites): record the CI evidence (section 3).
3. [Test Plan → Running Tests From TheRock Builds](#running-tests-from-therock-builds): prepare the artifacts and manifest (section 4), then run the tests and paste the output (section 5).
4. [Test Plan → Running Tests From Source Build](#running-tests-from-source-build): build from source and record its results (section 6).
5. Sign off once every plan step is evidenced (section 7).

---

### 1. Header / traceability

Fill in the identifiers that let a reader trace exactly what was validated back to source.

- **Feature or milestone**: `<feature name or milestone>`
- **Epic / ticket**: `<JIRA-EPIC>` / `<JIRA-TICKET>`
- **Delivery PR**: `<https://github.com/ROCm/rocm-libraries/pull/NNNN>`
- **PR head commit**: `<sha>`
- **PR merge commit**: `<https://github.com/ROCm/rocm-libraries/commit/sha>`
- **GPU / ASIC family used**: `<e.g. gfx942 (MI300), gfx950 (MI350)>`
- **Build(s) validated** (artifact identifiers/URLs, which encode OS / ASIC / ROCm version):
  - `<https://.../therock-dist-linux-<gpu-family>-<version>.tar.gz>`
  - `<https://.../therock-dist-linux-<gpu-family>-tests-<version>.tar.gz>`

> The artifact identifier encodes the OS, ASIC family, and ROCm version, so those no longer need separate fields. Still note the physical GPU/ASIC family the run executed on above, since a multi-arch artifact can target several.

---

### 2. What was validated

One paragraph describing the scope of the change and what this run exercises.

> `<One-paragraph summary: what the feature/milestone delivers and what behavior this validation confirms.>`

Bulleted test areas covered (delete rows that do not apply):

- **Frontend unit tests**: `<suites / behaviors covered>`
- **Backend unit tests**: `<suites / behaviors covered>`
- **GPU integration tests**: `<suites / behaviors covered>`
- **Sample validation**: `<sample name and flows exercised>`

---

### 3. Passing evidence

> Records the [Test Plan → Prerequisites](#prerequisites) step (CI is green).

Link the CI, superbuild, and release-artifact runs that back this validation.

| Evidence | Link | Status |
|---|---|---|
| hipDNN Superbuild CI | `<actions run URL or N/A>` | `<Passed / N/A because not triggered>` |
| Primary TheRock CI / package tests | `<actions run URL or N/A>` | `<Passed / N/A because not triggered>` |
| Sanitizer or nightly validation | `<actions run URL or N/A>` | `<Passed / N/A with approved reason>` |
| Final TheRock / release artifact run | `<actions run URL>` | `<Passed / TBD>` |
| Final sample run artifact / log | `<actions run URL or N/A>` | `<Passed / N/A because not applicable / TBD>` |

`N/A` is valid only when the validation is outside this run's scope. A workflow that should have run but was skipped is not passing evidence; record an explicit waiver and sign-off.

> **Scope note:** The command/output snippets in section 5 are *focused* validation runs (targeted gtest/ctest filters for the feature under test). The *full* hipDNN CTest suite evidence comes from the linked CI job. A complete release validation runs **both** the focused filters below **and** the full installed CTest suite on the selected artifacts.

---

### 4. Replication setup

> Records the "obtain a ROCm build" step of [Test Plan → Running Tests From TheRock Builds](#running-tests-from-therock-builds).

How to prepare the validated artifacts on a GPU node so a reader can reproduce section 5.

Use a GPU node matching the artifact family (e.g. `gfx94X-dcgpu` for MI300/MI325, `gfx950-dcgpu` for MI350/MI355). Download the `...-tests.tar.gz` variant (a superset of the plain distribution tarball that also carries the test executables); see [Obtaining ROCm](Building.md#obtaining-rocm) for the filename structure and version selection. Run from an empty working directory:

```bash
mkdir <feature>-validation
cd <feature>-validation

curl -O https://rocm.nightlies.amd.com/tarball-multi-arch/therock-dist-linux-<gpu-family>-tests-<version>.tar.gz

mkdir rocm-artifacts
tar -C rocm-artifacts -zxf therock-dist-linux-<gpu-family>-tests-<version>.tar.gz
```

Record what the manifest inside the tree reports: the ROCm version and the exact source commit the build came from. Run the following and paste the output here:

```bash
# ROCm version/package version, the TheRock build commit, and the CI run that produced the build.
grep -E '"the_rock_commit"|"github_run_id"|"rocm_version"|"rocm_package_version"' rocm-artifacts/share/therock/therock_manifest.json
# The rocm-libraries source commit (pin_sha) the build came from, with its submodule name for context.
grep -A3 '"submodule_name": "rocm-libraries"' rocm-artifacts/share/therock/therock_manifest.json | grep -E '"submodule_name"|"pin_sha"'
```

Example output:

```text
  "the_rock_commit": "d9551ec95643eb12cc1fdb81a95530e105539415",
  "github_run_id": "30227121935",
  "rocm_package_version": "7.15.0a20260727",
  "rocm_version": "7.15.0",
      "submodule_name": "rocm-libraries",
      "pin_sha": "bbb68174083e83e9465b60ecd8e20e4439a8e101",
```

> Record the above output from the manifest; the `pin_sha` confirm the delivery commit (section 1) is contained in the history of that SHA. This proves the feature is actually present in the validated build. The manifest lists a `pin_sha` for every submodule, so use the `rocm-libraries` entry specifically, not the first `pin_sha` in the file.

See the [Test Plan](#running-tests-from-therock-builds) for more on obtaining a ROCm build with the hipDNN test executables.

---

### 5. Test commands and expected output

> Records the "running the hipDNN tests" step of [Test Plan → Running Tests From TheRock Builds](#running-the-hipdnn-tests).

The core of the record: repeatable command → observed-output blocks. Paste the **actual** output you saw. Reference the real hipDNN test binaries under `rocm-artifacts/bin/` (e.g. `hipdnn_frontend_tests`, `hipdnn_backend_tests`, `hipdnn_public_frontend_tests`). Numeric timings and tensor values vary by GPU and run.

#### Example: focused gtest filter run

```bash
./rocm-artifacts/bin/<test_binary> \
  --gtest_filter="<Suite1*:Suite2*>" \
  --gtest_brief=1
```

Observed passing output:

```text
[==========] <N> tests from <M> test suites ran. (<t> ms total)
[  PASSED  ] <N> tests.
```

#### Example: GPU integration filter (plugin dir required)

```bash
HIPDNN_PLUGIN_DIR="$(pwd)/rocm-artifacts/lib/test_plugins/custom" \
./rocm-artifacts/bin/hipdnn_public_frontend_tests \
  --gtest_filter="<*IntegrationFeature*>" \
  --gtest_brief=1
```

Observed passing output:

```text
[==========] <N> tests from <M> test suites ran. (<t> ms total)
[  PASSED  ] <N> tests.
```

#### Example: installed CTest filter

```bash
HIPDNN_PLUGIN_DIR="$(pwd)/rocm-artifacts/lib/hipdnn_plugins/engines" \
ctest --test-dir ./rocm-artifacts/bin/hipdnn_samples \
  -R <test_name_regex> \
  --output-on-failure
```

Observed passing output:

```text
100% tests passed, 0 tests failed out of <N>
```

> Repeat one command/output block per focused area from section 2. For the full-suite requirement in the scope note, run the installed hipDNN CTest suite (see the [Test Plan](#running-the-hipdnn-tests)) and record its `100% tests passed, 0 tests failed out of <N>` line here as well.

---

### 6. Optional: build from source against installed artifacts

> Records the [Test Plan → Running Tests From Source Build](#running-tests-from-source-build) step. Reuse the `rocm-artifacts` tree from section 4 as the build's ROCm dependency, as the plan describes.

Only needed when validating the **source** itself, or when the artifact ships runtime/devel packages but no prebuilt test/sample binaries. This builds against the installed artifacts using a user-supplied toolchain, so paths and the CMake prefix are explicit rather than preset-driven.

Check out the matching source tree (use the `pin_sha` from section 4, or `develop` after merge):

```bash
git clone --filter=blob:none --no-checkout https://github.com/ROCm/rocm-libraries.git rocm-libraries
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/hipdnn dnn-providers cmake test
git checkout <develop-or-pin-sha>
```

Configure and build against the installed artifacts (run from the `rocm-libraries` folder, sibling to `rocm-artifacts`):

```bash
cmake -S projects/hipdnn \
  -B build/hipdnn-tests \
  -GNinja \
  -DCMAKE_PREFIX_PATH="$(pwd)/../rocm-artifacts/lib/cmake" \
  -DROCM_PATH="$(pwd)/../rocm-artifacts" \
  -DCMAKE_C_COMPILER="$(pwd)/../rocm-artifacts/lib/llvm/bin/clang" \
  -DCMAKE_CXX_COMPILER="$(pwd)/../rocm-artifacts/lib/llvm/bin/clang++" \
  -DHIPDNN_SKIP_TESTS=OFF \
  -DENABLE_CLANG_TIDY=OFF

cmake --build build/hipdnn-tests --target <targets>
ctest --test-dir build/hipdnn-tests --output-on-failure --parallel 8
```

> To build a sample instead, point `-S` at `projects/hipdnn/samples` and build the sample target. Expected result: the same passing snippets as section 5, and a full-suite `100% tests passed, 0 tests failed out of <N>`.
>
> This flow builds against the installed artifacts with an explicit toolchain and CMake prefix. For an ordinary standalone source build using the checked-in configure presets, follow the [Quick Start Guide](Building.md#quick-start-guide) instead.

---

### 7. Final validation checklist

Sign-off gate. Every box must be checked with real evidence before marking the milestone verified.

- [ ] Filled section 1 with the merged release/nightly **build identifier(s)** and the physical GPU/ASIC family used.
- [ ] Confirmed the delivery commit is contained in the artifact's manifest `pin_sha` (section 4).
- [ ] Replaced every `TBD` in section 3 with final evidence or an explicit `N/A` reason; included sample evidence when samples are in scope.
- [ ] Ran the applicable focused gtest/ctest filters **and** the full installed hipDNN CTest suite on the final artifacts.
- [ ] Pasted the passing output for every executed scope into section 5.
- [ ] Confirmed all [Test Plan](#release-and-milestone-test-plan) prerequisites (CI green, documentation/changelog current) are satisfied.

**Tips for a defensible record:**

- Paste exact output, including failing messages if any; do not summarize away detail.
- Note environmental factors (GPU family, driver/ROCm version) that could affect results; the artifact identifier captures most of this.
- If a test consistently skips (e.g. GPU-gated or ASAN-incompatible), record why rather than deleting it. See [Environment](Environment.md#environment-variables) for enabling logging when a run needs deeper insight.

---

## Developer Testing Commands and Authoring Reference

This document is the developer reference for running and writing hipDNN tests: local commands, CTest/check targets, build requirements, categories, naming, and authoring patterns. For current validation governance, CI signals, ownership, supported configurations, release procedures, and known gaps, see the project [Testing Strategy](#hipdnn-testing-strategy).

### Running Tests

Both the superbuild and the standalone build follow the same three-step flow using cmake presets:

- `cmake --preset <configure-preset>`
- `cmake --build <binaryDir>`
- `ctest --test-dir <binaryDir> --output-on-failure` (or a `ninja` check target)

#### Superbuild (root `CMakePresets.json`)

Run from the repository root. binaryDir is `build`. The toolchain is baked into these presets; hipDNN developers may prefer to override it with the hipDNN toolchain (see [Building § Superbuild](Building.md#superbuild)).

For the hipDNN-relevant configure presets and the components each one enables, see the preset table in [Building § Superbuild](Building.md#superbuild).

> [!IMPORTANT]
> Root-level `ctest` is gated by `ROCM_LIBS_ENABLE_ROOT_CTEST`, which defaults **OFF** and is not set by any checked-in preset. Enable it, otherwise `ctest --test-dir build` sees no tests. Set it with `-DROCM_LIBS_ENABLE_ROOT_CTEST=ON` at configure time, or in the environment before the first (or a fresh) configure; once configured, the value is stored in the CMake cache and reused.

```bash
# From the repository root
cmake --preset hipdnn-dev-all -DROCM_LIBS_ENABLE_ROOT_CTEST=ON
cmake --build build
ctest --test-dir build # --output-on-failure is optional
```

Equivalently, set it in the environment. The variable is only read on a first or fresh configure, so use `--fresh` to force it to take effect on an already-configured build directory:

```bash
# From the repository root
export ROCM_LIBS_ENABLE_ROOT_CTEST=ON
cmake --preset hipdnn-dev-all --fresh
cmake --build build
ctest --test-dir build # --output-on-failure is optional
```

Superbuild ninja check targets are prefixed with `hipdnn-` (the bare, unprefixed aliases are not created in superbuild):

- `hipdnn-check`
- `hipdnn-quick-check`
- `hipdnn-unit-check`
- `hipdnn-integration-check`
- each also has a `-verbose` variant (e.g. `hipdnn-check-verbose`)

#### Standalone (`projects/hipdnn/CMakePresets.json`)

Run from the `projects/hipdnn/` directory.

> [!NOTE]
> The standalone build defaults to the `cmake/ClangToolChain.cmake` toolchain, which auto-detects the ROCm Clang compiler from your PATH (via `hipconfig`). If ROCm is not on your PATH, point CMake at it with `-DROCM_CMAKE_PATH=<rocm-root>`. See [Building § ROCM_PATH, ROCM_CMAKE_PATH, and CMAKE_INSTALL_PREFIX](Building.md#rocm_path-rocm_cmake_path-and-cmake_install_prefix) for the full toolchain-discovery details.

```bash
# From projects/hipdnn/
cmake --preset debug
cmake --build build/debug # binaryDir is `build/debug`
ctest --test-dir build/debug # --output-on-failure is optional
```

Standalone creates **unprefixed** aliases in addition to the prefixed targets:

- `check` (alias of `hipdnn-check`)
- `quick-check`
- `unit-check`
- `integration-check`
- plus the prefixed `hipdnn-*-check` names and their `-verbose` variants

#### Address Sanitizer

Add `-DBUILD_ADDRESS_SANITIZER=ON` to a standalone configure step, then run `ctest -L standard`; see [Building § Address Sanitizer Build](Building.md#address-sanitizer-build) for platform prerequisites and commands.

The current trusted ASAN signal is a manually run standalone check. A Linux pull-request workflow exists but is opt-in and currently builds sanitizer variants without running their tests; a scheduled Linux full-ASAN workflow also exists but is not yet an owned hipDNN signal. Windows ASAN is manual and has no CI lane. See [Testing Strategy § ASAN, TSAN, and Sanitizer Coverage](#asan-tsan-and-sanitizer-coverage) for current scope, exclusions, and tracked gaps.

### ctest vs. check targets

The key difference is what each one does with the test binaries:

- **A check target (`ninja hipdnn-check`) builds the tests first, then runs them.** After editing source, it recompiles what changed and runs the result, so you never test a stale binary. This is the everyday choice while developing.
- **`ctest` runs already-built tests and does not build anything.** It is faster when nothing has changed, but you must have built the tests first, and it gives full control over which tests run and how.

Use a check target when you want to build-and-run a canned scope in one step. Use `ctest` directly when the tests are already built and you want to filter, parallelize, or repeat the run (see the flags below).

Each check target runs a fixed `ctest` invocation (for the full catalog of build and test targets including the provider and superbuild targets, see [Building § Build Targets](Building.md#build-targets)):

| ninja target (superbuild) | Equivalent ctest command |
|---------------------------|--------------------------|
| `hipdnn-check` | `ctest --output-on-failure` |
| `hipdnn-quick-check` | `ctest -L quick --output-on-failure` |
| `hipdnn-unit-check` | `ctest -L unit --output-on-failure` |
| `hipdnn-integration-check` | `ctest -L integration --output-on-failure` |

Check targets **hard-code their flags**; there is no pass-through. Use a raw `ctest --test-dir <binaryDir>` invocation when you need:

- `-R <regex>` - filter (include) tests by name
- `-E <regex>` - filter (exclude) tests by name
- `-L <regex>` - filter (include) tests by label (category)
- `-LE <regex>` - filter (exclude) tests by label (category)
- `-j <N>` - run tests in parallel
- `--repeat until-fail:<N>` - flake hunting

> [!TIP]
> Running `ctest --test-dir <binaryDir>` directly drops the check target's baked-in `-L <category>`. To reproduce a category's filtering, re-add the label yourself (e.g. `ctest --test-dir build -L quick -j 8` matches `hipdnn-quick-check` plus parallelism).

### What you need to build

hipDNN **core tests need no real provider.** Every plugin the core suites load is an in-tree fake from `tests/test_plugins/` (linking only the SDKs + `hip::host`), wired in via `add_dependencies`. A provider-free build (standalone or the core-only `hipdnn` superbuild preset) fully exercises the core suites.

Real-provider and cross-project integration suites (`hipdnn_integration_tests`, `miopen_plugin_*`) live under `dnn-providers/`. They are opt-in components and are not required to run core tests.

Guidance:

| Goal | Use |
|------|-----|
| Full validation (all providers + integration + samples) | `hipdnn-dev-all` superbuild preset |
| Everyday core work | `hipdnn` (core-only superbuild preset) |
| Minimal / offline core loop | standalone (`cmake --preset debug`) |

### Quick Reference

#### Test Organization

| Component | Test Location | Type |
|-----------|--------------|------|
| Backend | `backend/tests/` | Unit tests |
| Frontend | `frontend/tests/` | Unit tests |
| Data SDK | `data_sdk/tests/` | Unit tests |
| FlatBuffers SDK | `flatbuffers_sdk/tests/` | Unit tests |
| Plugin SDK | `plugin_sdk/tests/` | Unit tests |
| Test SDK | `test_sdk/tests/` | Unit tests |
| Backend logging | `tests/backend_logging/` | Unit tests |
| API | `tests/backend/` | Black box API tests |
| Frontend integration | `tests/frontend/` | Integration tests |

> [!NOTE]
> Provider tests live under `dnn-providers/` and are built only when the corresponding provider component is enabled.

#### Test Naming

Every GoogleTest test has two parts: a **suite name** and a **case name**, written `SuiteName.CaseName`. These are the two arguments to the test macro: in `TEST(TestGpuConvolutionFp16, ForwardProducesExpectedOutput)`, the suite name is `TestGpuConvolutionFp16` and the case name is `ForwardProducesExpectedOutput`. A convention on both, checked by the `hipdnn-validate_test_names` target (`cmake/scripts/test_name_validator.py`), requires the suite name to be:

```
(Test|Integration)[Gpu]FeatureName[Datatype]
```

- Both names are PascalCase and may not contain `_ ; : < > [ ] ,`.
- The suite name starts with `Test` or `Integration`, optionally followed by `Gpu`, then the feature name, optionally followed by a datatype (`Fp16`, `Fp32`, `Fp64`, `Bfp16`).
- Keywords (`Gpu`, the datatypes, and the shape keywords `Nhwc`, `Nchw`, `Ndhwc`, `Ncdhw`) belong in the **suite** name, not the case name, must use exactly that capitalization, and should appear only once.

For example, `TestGpuConvolutionFp16.ForwardProducesExpectedOutput` is valid. Invalid: `Test_conv` (underscore), `TestConvolutionFp16.RunsFp16` (datatype keyword in the case name), and `TestConvolutionFp16Fp16` (keyword repeated).

#### CTest Categories

Categories are defined in `test_categories.yaml`. There are six:

- `quick` - pattern matches all tests
- `unit` - selects named unit suites directly
- `integration` - selects named integration suites directly
- `standard`, `comprehensive`, `full` - no direct patterns; they inherit `quick`'s tests via tier labels

The tiers are cumulative: `standard` includes everything in `quick`, `comprehensive` includes everything in `standard`, and `full` includes everything in `comprehensive`. `unit` and `integration` are not tiers; they select their own named suites independently.

Each category is exposed as a ctest label, so you can filter by it directly: `ctest -L <category>` runs a category (and `-LE <category>` excludes one) - the same label the matching `hipdnn-<category>-check` target uses. See [ctest vs. check targets](#ctest-vs-check-targets).

> [!NOTE]
> Today the higher tiers add no tests of their own, so `quick`, `standard`, `comprehensive`, and `full` all run the same set. The distinction exists for when higher tiers gain their own tests later.

#### Choosing Between TYPED_TEST and TEST_P

```
Need to test across multiple data types (float, half, bfloat16)?
│
├── NO → Use TEST() or TEST_F()
│
└── YES → Also need parameterized test cases?
          │
          ├── NO → Use TYPED_TEST
          │        (Compile-time type safety, simple)
          │
          └── YES → Use TEST_P with multi-declarations
                    (Handles both types AND parameters)
```

**Key Principle:** Prefer `TEST_P` over `TYPED_TEST` when both type variation and parameterized cases are needed. `TYPED_TEST` and `TEST_P` don't mix well together.

| Scenario | Approach |
|----------|----------|
| Single type, no params | `TEST()` / `TEST_F()` |
| Single type, with params | `TEST_P()` |
| Multi-type, no params | `TYPED_TEST` |
| Multi-type, with params | `TEST_P` + multi-declarations |

#### Multi-Declaration Pattern (for types + parameters)

When you need both type variation AND parameterized tests, use explicit type aliases with `TEST_P`:

```cpp
template <typename DataType>
class ConvTest : public ::testing::TestWithParam<ConvTestCase> { };

// Explicit type aliases - one per type
using ConvTestFp32 = ConvTest<float>;
using ConvTestFp16 = ConvTest<half>;
using ConvTestBfp16 = ConvTest<hip_bfloat16>;

// Separate TEST_P for each type
TEST_P(ConvTestFp32, Correctness) { runTest(); }
TEST_P(ConvTestFp16, Correctness) { runTest(); }
TEST_P(ConvTestBfp16, Correctness) { runTest(); }

// Separate instantiation for each type
INSTANTIATE_TEST_SUITE_P(Smoke, ConvTestFp32, testing::ValuesIn(getCases()));
INSTANTIATE_TEST_SUITE_P(Smoke, ConvTestFp16, testing::ValuesIn(getCases()));
INSTANTIATE_TEST_SUITE_P(Smoke, ConvTestBfp16, testing::ValuesIn(getCases()));
```

**Why multi-declarations over macros?** Modern tooling makes boilerplate easy to handle, and avoiding macros makes tests easier to debug and understand.

#### Type Combinations with TypePair

For testing type combinations (e.g., input type + compute type), define a struct to hold the types:

```cpp
template <typename T1, typename T2>
struct TypePair
{
    using InputType = T1;
    using ComputeType = T2;
};

using TypeCombinations = ::testing::Types<TypePair<float, float>,
                                          TypePair<half, float>,
                                          TypePair<hip_bfloat16, float>>;
TYPED_TEST_SUITE(MyTypedTest, TypeCombinations);

TYPED_TEST(MyTypedTest, Correctness)
{
    using InputType = typename TypeParam::InputType;
    using ComputeType = typename TypeParam::ComputeType;
    // Test implementation using InputType and ComputeType
}
```

#### Testing Requirements

- **Coverage Target**: 80% overall is the goal, with each component aiming for >80% individually (a target, not machine-enforced)
- **GPU Tests**: Must be marked with the `SKIP_IF_NO_DEVICES()` macro
- **Platform Support**: Tests target Windows and Linux, but automated configurations and platform exclusions differ; see the [validated configuration matrix](#supported-and-validated-configurations)
- **Performance**: Unit tests must execute quickly
- **CI**: All applicable triggered checks must be inspected; a skipped workflow is not passing evidence

### Expectations During Development

Tests are a merge gate, not an afterthought. The following expectations apply to every PR.

**MUST:**

- Defect fixes ship a regression test proven to fail before / pass after the fix - locks the bug so it can't silently return.
- Product-code changes carry a test, a safe-default flag, or a written waiver - no behavior change ships unverified.
- Never disable/skip/weaken a test to green CI (no waiver) - greening by removing coverage hides real failures.
- Tests must assert real behavior - a test no source change could break is coverage padding, not coverage.
- ASAN/leak-clean by design - Linux full-ASAN runs nightly and standalone checks are available, but no sanitizer test currently gates pull requests; write RAII / failure-path cleanup and record any sanitizer validation actually run.
- PR body carries an honest Testing Summary + Checklist - `[x]` only for validation that actually passed, with the exact command.

**SHOULD:**

- Record the "why" for non-obvious tolerances or parameter sets - prevents unmaintainable orphaned tests.
- Follow existing test naming/placement so the suite actually picks the test up.
- Provider/support-surface changes (`dnn-providers/**`) need multi-arch coverage - a family that builds but skips tests is uncovered, not covered.
- Verify negative/error paths - unsupported op/layout/dtype combinations must fail predictably, not run a wrong path.

### Testing Strategy and Governance

The authoritative component strategy is [TESTING.md](#hipdnn-testing-strategy). It records the current validation model, testing layers, CI classifications, coverage scope, supported configurations, sanitizer status, ownership handoffs, release procedures, improvement roadmap, and known risks and gaps.

The [Detailed Testing Layer Reference](#detailed-testing-layer-reference) covers white-box unit, black-box API, and end-to-end integration layers. The Quick Reference patterns above remain the day-to-day authoring guide.

### Release / Milestone Verification

The following sections of [TESTING.md](#hipdnn-testing-strategy) cover **hipDNN milestone and release verification**: validating a release build and capturing evidence for sign-off. They are **not** part of the per-PR development workflow; day-to-day contributors do not need them (see [Expectations During Development](#expectations-during-development) above for the PR-time bar).

- [Release and Milestone Test Plan](#release-and-milestone-test-plan) - release verification checklist and expected results.
- [Test Run Record Template](#test-run-record-template) - release validation evidence template.
