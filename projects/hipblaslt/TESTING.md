# hipBLASLt Testing Strategy

**Status:** Draft
**Owner:** Tony Davis (@tony-davis)
**Technical Lead:** TBD before this document leaves draft
**Last Updated:** 2026-07-31

This document describes how hipBLASLt is tested today, which signals actually gate a merge, and
where the holes are. It follows the ROCm-wide TESTING.md template.

It is deliberately written as an accurate description of the current state rather than an
aspirational one. Several sections record gaps. Recording a gap is the purpose of the exercise, not
a failure of it, and a gap that is written down is one we can argue about and close.

For the mechanics of *how* the client test flow works (YAML data files, `gentest` expansion, GTest
registration), see the test documentation landing in
[PR #10205](https://github.com/ROCm/rocm-libraries/pull/10205). This document is the strategy layer
and does not duplicate it.

## Component Overview

hipBLASLt is a general matrix-multiply (GEMM) library exposing a flexible API with extended
functionality: fused epilogues, mixed and low-precision datatypes, block-scaled formats, grouped
GEMM, and user-driven kernel tuning. It sits in the math-libraries layer of ROCm, above HIP and the
ROCr runtime, and underneath frameworks such as PyTorch. See [README.md](README.md) for the API
surface and supported operations.

Two things about the architecture shape everything below.

**Most of the product is generated code.** The GEMM kernels are not hand-written C++ in this
repository. They are emitted by TensileLite, a Python code generator under
[`tensilelite/`](tensilelite/), which produces assembly kernels and the solution-selection library
that the C++ runtime loads at dispatch time. So hipBLASLt has two distinct codebases with two
distinct test strategies: a Python generator that is highly unit-testable, and a C++ runtime that
is almost entirely GPU-coupled.

**Almost all C++ behavior is only observable on a GPU.** Correctness of a GEMM means numerical
agreement with a reference on real hardware, for a specific architecture, datatype, and problem
shape. Very little of the C++ library can be meaningfully validated without a device, and the parts
that could be are currently entangled with the parts that cannot (see
[Unit Testing Strategy](#unit-testing-strategy)).

The practical consequence is that hipBLASLt's confidence comes overwhelmingly from large
integration suites on real hardware, and the unit-test story is strongest in exactly the place most
people do not look first: the Python code generator.

Major dependencies: HIP and the ROCm compiler toolchain, the ROCr runtime, `rocisa` (the Python
extension backing TensileLite instruction generation), and the shared build and CI infrastructure
provided by TheRock.

## Development Workflow

What a developer does between making a change and getting it merged.

**1. Build.** Follow [README.md](README.md) for a standalone build, or build within TheRock
superbuild. Client binaries (`hipblaslt-test`, `hipblaslt-bench`) are built by default when
`HIPBLASLT_ENABLE_CLIENT=ON` and `HIPBLASLT_BUILD_TESTING=ON`, which are both the default.

**2. Run the tests that match what you touched.**

| You changed | Run this | Needs a GPU |
| --- | --- | --- |
| C++ library or client code | `hipblaslt-test --gtest_filter=*smoke*` for a fast check, then a wider tier | Yes |
| TensileLite Python | `cd tensilelite && tox -e unit` | No |
| `rocisa` | `cd tensilelite && tox -e rocisa` | No |
| TensileLite kernel-generation behavior | `tox -e unit` (includes the characterization goldens) | No |
| Anything, before pushing | `pre-commit run --all-files` | No |

The four client test tiers (`quick`, `standard`, `comprehensive`, `full`) are defined in
[`clients/tests/test_categories.yaml`](clients/tests/test_categories.yaml). When hipBLASLt is built
inside rocm-libraries, those tiers are registered as CTest labels and a relocatable
`CTestTestfile.cmake` is installed to `bin/hipblaslt/`, so the tiers can be run with `ctest` from
that directory. In a standalone or sparse checkout the CTest categorization is skipped and you
select tests with `--gtest_filter` directly.

**3. Add the right kind of test.** See [Choosing the Right Test Type](#choosing-the-right-test-type).
For a bug fix, the expectation is a regression test that fails before your fix and passes after it.

**4. Open the PR** following [CONTRIBUTING.md](CONTRIBUTING.md), targeting `develop`. Conventional
Commits title, and a reference to the issue or ticket the work belongs to.

**5. Watch the right checks.** A PR touching `projects/hipblaslt/**` triggers several independent
lanes described in [Pre-submit / CI Gates](#pre-submit--ci-gates). The signal you need before
asking for review is a green build plus a green quick-tier test run; the multi-architecture suites
take considerably longer.

## Testing Strategy and Layers

### Unit Testing Strategy

**Purpose.** Validate logic that can be exercised without dispatching a kernel to a physical
device: argument validation, kernel and solution selection, workspace sizing, data-structure
construction, type and format conversion, and error propagation.

The ROCm stack was not architected with hardware-independent testability in mind, and the practical
amount of unit-testable logic varies significantly by component. hipBLASLt is a clear example of
that variance, because its two codebases sit at opposite ends of the spectrum.

#### TensileLite Python (strong)

This is where hipBLASLt's real unit testing lives. The code generator is pure Python producing
text, so essentially all of it is testable with no GPU.

| Item | Detail |
| --- | --- |
| Framework | pytest, orchestrated by `tox` |
| Location | [`tensilelite/Tensile/Tests/unit/`](tensilelite/Tensile/Tests/unit/) |
| Golden snapshots | [syrupy](https://github.com/syrupy-project/syrupy), `.ambr` files in per-module `__snapshots__/` directories |
| How to run | `cd tensilelite && tox -e unit` (skips the client build), or `tox -e rocisa` for the extension only |
| Coverage measurement | `tox -e coverage-unit` |

A significant part of that suite is a **characterization suite** under
[`tensilelite/Tensile/Tests/unit/characterization/`](tensilelite/Tensile/Tests/unit/characterization/),
established in [PR #7989](https://github.com/ROCm/rocm-libraries/pull/7989) and grown considerably
since. It now spans more than 60 characterized modules across roughly 250 test files, backed by
98 `.ambr` golden files, plus a separate codegen harness that characterizes generated assembly
per architecture. These are explicitly *not* specification tests: they pin down what the code does
today, including latent bugs, so that the ongoing consolidation refactor shows any unintended
behavior change as a reviewable diff rather than a silent downstream regression. Latent bugs found
while characterizing are flagged in that directory's `DECISIONS.md` rather than silently fixed, so a
golden that encodes wrong behavior is documented as such.

The discipline that goes with goldens is documented in the suite's
[README.md](tensilelite/Tensile/Tests/unit/characterization/README.md) and is worth stating here
because it is easy to get wrong: **never run a blanket `pytest --snapshot-update`.** It rewrites
every golden at once and produces a green run that proves nothing. Update the smallest node id you
intend to change, read the resulting diff, and explain the behavior change in your PR description.

**The goldens are enforced.** Any pull request touching `projects/hipblaslt/tensilelite/**` runs the
`Component CI: TensileLite coverage` GitHub Actions lane, which executes `tox -e coverage-unit` over
the characterization tree with syrupy installed. A stale golden fails that check. The same tox
environment also runs in Math CI, so there are two independent lanes asserting the same snapshots.

There is one place snapshots do *not* run: the installed-artifact lane, which re-runs the unit tree
from the packaged `share/hipblaslt/tensilelite/` install. That environment does not ship syrupy, so
the suite's `conftest.py` detects the missing plugin and skips the snapshot-using tests cleanly
rather than erroring the whole run. Nothing is lost, because those goldens were already asserted in
the source lanes before the artifact was built, but a reader scanning that run will see skips and
should know they are deliberate.

The suite also carries two enforced coverage floors: a whole-project floor and a per-file ratchet
with a one percentage point tolerance, so per-file coverage can only move up over time. Both are
enforced in CI (see [Coverage](#coverage)).

A GPU-less seam, the `--cpu-only` switch, lets the client and device-probe paths of the benchmark
flow be exercised without hardware, via an architecture spoof and a client-launch stub. Its
performance output is synthetic and fixed, so it is useful for testing the plumbing and useless for
anything performance-related. The switch is covered by
[`tensilelite/Tensile/Tests/unit/test_cpu_only_switch.py`](tensilelite/Tensile/Tests/unit/test_cpu_only_switch.py)
and documented, with that caveat, at
[`_codegen/GPU-MOCK.md`](tensilelite/Tensile/Tests/unit/characterization/_codegen/GPU-MOCK.md).

**Mutation testing** is a report-only pilot on a small slice of modules, run through
`tox -e mutation-unit` and configured in `pyproject.toml`. It is not a gate and does not run in CI.
A series of PRs widening the mutation-hardened surface starts at
[PR #10133](https://github.com/ROCm/rocm-libraries/pull/10133); those are still in draft.

#### C++ library and clients (weak, and structurally blocked)

There is no separate unit-test binary. The host-only tests that do exist are compiled into
`hipblaslt-test` alongside the GPU tests, in
[`clients/tests/src/`](clients/tests/src/):

| File | Suites | What it covers |
| --- | --- | --- |
| `secure_env_gtest.cpp` | `SecureEnv` | Environment-variable policy for privileged processes |
| `benchmark_stats_gtest.cpp` | `ValidateAdaptiveConfig`, `RotatingBufferPlan`, `RatePerSecond` | Adaptive-benchmark statistics helpers |
| `caching_library_gtest.cpp` | `CachingLibraryCollision` | Solution-cache keying under a forced hash collision |

That is 30 test cases of genuinely hardware-independent logic, against a library of substantial
size. It is not a meaningful unit-test suite; it is three useful islands.
[PR #9733](https://github.com/ROCm/rocm-libraries/pull/9733) is in flight and adds the first
dedicated host-only unit target along with a `HostUnit*` suite-naming convention, covering the
client datatype and scaling-format helpers.

The reason the C++ side is thin is structural rather than cultural, and worth naming precisely so
the fix is obvious:

1. **Internal headers are not reachable from the test target.** Adding
   `library/src/amd_detail/rocblaslt/src/include` to `hipblaslt-test`'s include path breaks the
   build, because it shadows the client's own `utility.hpp`. The workaround in use today is a
   relative `#include` of a specific library header from a specific test file, which does not scale.
2. **There is no host-only link target.** The library builds as a single shared object, so a unit
   test cannot link just the host-side pieces. A white-box path does exist
   (`HIPBLASLT_ENABLE_COVERAGE=ON` links the library's object files into `hipblaslt-test`), but it is
   not the default and is intended for coverage rather than for unit testing.
3. **"Host-only" still means "compiled as HIP."** Client headers require HIP compilation because the
   float8 support headers do not behave in a plain C++ translation unit.
4. **Validation is interleaved with dispatch.** Argument checks frequently sit inside functions that
   have already allocated device memory or are about to launch a kernel, so they cannot be reached
   without a GPU.
5. **Configuration is read through singletons.** Logging and tuning-override state initialize from
   the environment on first use with no reset hook, making env-dependent behavior order-sensitive
   under test.

None of these are hard problems. They are the roadmap (see
[Improvement Roadmap](#improvement-roadmap)).

**Explicitly not covered by unit tests:** GEMM numerical correctness, kernel selection against real
hardware, anything in the generated assembly, memory-management behavior, stream and event
semantics, and multi-GPU behavior. All of that is integration territory by construction.

**Coverage expectation.** For TensileLite Python, the target is 80% patch coverage on modified code,
with the project as a whole expected to reach 80% by end of 2026. Current whole-project Python
coverage is far below that, which is exactly why the floors are ratchets rather than a fixed bar.
For the C++ side there is no coverage target today, because with the current structure a target
would be aspirational rather than actionable.

### Integration Testing Strategy

**Purpose.** Validate what unit tests cannot: numerical correctness of real GEMMs on real hardware,
across the datatype, epilogue, and shape space the library claims to support.

This is where nearly all of hipBLASLt's confidence comes from.

**What is covered.** The client test suite (`hipblaslt-test`) is a GTest binary driven by YAML data
files in [`clients/tests/data/`](clients/tests/data/). The YAML describes problem configurations
(datatypes, transposes, dimensions, leading dimensions, alpha and beta, epilogues, scaling,
initialization patterns) which are expanded at build time into individual GTest cases. The bulk of
the space lives in `matmul_gtest.yaml` and `hipblaslt_common.yaml`. Coverage includes matmul across
datatypes and epilogues, grouped GEMM, matrix transform, auxiliary and descriptor APIs, extension
operations such as layernorm and amax, ULP-level numerical checks, and Stream-K.

Each case validates against a CPU reference computation with tolerance checks, so a failure means a
real numerical disagreement rather than a crash only.

**What requires GPU hardware:** effectively all of the above.

**What can run CPU-only:** the TensileLite Python suites, `rocisa`, the host-only cases listed in
the previous section, and the kernel-config build phase (`test_config_build.py`), which compiles
kernels on CPU and hands an artifact to a GPU stage for the run phase.

**Tiers and duration.** Defined in
[`clients/tests/test_categories.yaml`](clients/tests/test_categories.yaml):

| Tier | Contents | Target duration | Timeout |
| --- | --- | --- | --- |
| `quick` | smoke | ~5 min | 600 s |
| `standard` | smoke + quick + pre_checkin | ~30 min | 3600 s |
| `comprehensive` | standard + nightly | ~2 h | 7200 s |
| `full` | comprehensive + HMM (needs a managed-memory capable host) | up to 24 h | 86400 s |

All tiers exclude `*known_bug*`. There are currently no multi-GPU tests.

**A caveat on how tiers are actually applied.** The TheRock test lane does not invoke CTest; it runs
the `hipblaslt-test` binary directly with a GTest filter. In that lane, `quick` maps to `*smoke*`,
and every tier above `quick` currently applies no filter at all and runs the entire binary. So the
four-tier taxonomy above is honored by CTest but only half-honored by the lane that runs in CI. See
[Known Risks and Gaps](#known-risks-and-gaps).

**Test size and shape expectations.** The suite leans heavily on many small problem sizes rather
than few large ones, which is the right default: most correctness bugs are in tiling, edge handling,
and epilogue fusion, and those reproduce at small sizes far more cheaply. Large shapes matter for
address-arithmetic overflow and workspace behavior specifically, and are used sparingly for that
reason. There is real redundancy in the datatype sweeps, where many numerical variants exercise the
same code path, and pruning that is a standing opportunity rather than an active project.

**Pre-flight layout validation.** Before the GTest binary runs, the TheRock lane walks the installed
tree and validates its physical layout. This exists because the runtime's kernel-library probe has
no fallback for a misplaced file, so a packaging regression would otherwise surface as an opaque
`hipModuleLoad` failure at first dispatch instead of a clear error naming the offending path.

**What runs only during release qualification:** the `full` tier, including the HMM cases, and
broader hardware coverage than PR or nightly CI provides. See
[Release Validation](#release-validation).

### Performance and Benchmarking Testing

hipBLASLt is a performance library, so this section deserves to be read carefully: the tooling is
good, and the automation is absent.

| Item | Detail |
| --- | --- |
| Stack layer | Core SDK (math library) |
| Metrics measured | GFLOP/s, memory bandwidth (GB/s), latency (µs). Optionally clock frequencies, achieved efficiency, and memory read/write bytes |
| How benchmarks are run | `hipblaslt-bench` for a single problem or a YAML batch; `hipblaslt-perf` to run named suites and aggregate CSV over repeated samples; `hipblaslt-cotenant` to measure under CU contention |
| Baseline stored per architecture | **No.** No per-architecture baselines are stored in this repository |
| Where results are stored | Local CSV in the developer's workspace. Optional export of a commit hash and machine specification for external tracking. Nothing is persisted by CI |
| Regression threshold | **None defined.** The tuning flow uses an uplift threshold to decide whether to keep a tuned kernel, which is a candidate filter, not a regression gate |
| Gating approach | Manual, ad hoc |

**Gating, stated honestly:**

| Gating level | Status | Notes |
| --- | --- | --- |
| PR-level automated gate | **No** | Known gap. Nothing in CI runs a benchmark and compares it to anything |
| Nightly automated comparison | **No** | Scaffolding for a last-known-good comparison exists in TheRock's extended tests, but it is currently marked expected-to-fail and its results API is disabled |
| Manual nightly review | **Informal** | Developers and tuning owners run benchmarks by hand around changes they believe are performance-sensitive. There is no scheduled review and no named reviewer |
| Release qualification | **No documented gate** | Performance is discussed at release time but there is no in-repo criterion |

There is a function named `ci_perf_job` in the performance suite definitions, documented as "run
basic job for PR-CI". It has no callers anywhere in the repository. Mentioning it here so nobody
concludes from its name that a performance gate exists.

Related tooling that is sometimes mistaken for regression testing: `utilities/geko/` is a GEMM
kernel optimizer that searches for better kernels and benchmarks candidates during tuning, and
`utilities/QuickTune/` performs offline per-workload tuning from captured logs. Both produce
performance numbers. Neither compares against a stored baseline, and neither runs in CI.
Similarly, TensileLite's benchmark and library-logic phases generate performance data used to
*select* kernels during tuning; that data is not retained as a regression baseline.

**Known gaps:** no stored per-architecture baselines, no threshold, no automated comparison at any
cadence, no dashboard, and no gate on library size, kernel count, or build time. A performance
regression introduced by a code or tuning change is currently caught by a human noticing, or by a
downstream consumer.

## Pre-submit / CI Gates

hipBLASLt is validated by two CI systems, which is the single most important thing to understand
about its signal. GitHub Actions runs the build and the shared TheRock-based test lanes. A separate
internal Jenkins-based system ("Math CI") runs the multi-architecture TensileLite suite and posts
checks named `mci/rocm-libraries/...` onto the same PR. Both appear in the PR checks list.

### Validation Gates and Ownership

| Validation area | Required before merge | Owner | Responsibility |
| --- | --- | --- | --- |
| Build (Linux and Windows) | Yes | CI / DevOps + TheRock | Maintain jobs, runners, and pipeline health |
| Unit tests (TensileLite Python) | Yes | Component team | Create, maintain, review |
| Integration / smoke tests (client GTest) | Yes | Component team | Validate behavior across key scenarios |
| Characterization goldens | Yes, when `tensilelite/` is touched | Component team | Review every golden diff; never bulk-regenerate |
| HOST_ASAN build and quick test | Yes, on gfx90a | Component team | Keep the sanitizer lane green |
| Code coverage floor and ratchet (TensileLite) | Yes, when `tensilelite/` is touched | Component team / CI | Floors move up only |
| Formatting and lint (`pre-commit`) | Yes | CI / DevOps | Maintain hooks |
| Static analysis | No | CI / DevOps | Not currently run for hipBLASLt |
| Performance | No | Component team | No gate exists (see above) |
| Shared validation infrastructure | N/A | TheRock team | Provide shared build and test infrastructure |
| System validation | N/A | QA | Execute system-level and release validation |
| Release qualification | N/A | Component team + QA + TPM | Confirm readiness, review known risks |

A caveat on this table: which checks are configured as *required* in branch protection is not
documented anywhere a contributor can see, and the answer is not consistently understood even among
the people working on the tests. The rows above reflect intent and observed behavior. Confirming
and publishing the actual required-check list is tracked as a gap.

### PR Test Classification

| Status | Applies to |
| --- | --- |
| **Trusted gate** | Build (both platforms); client GTest quick tier on gfx94X-dcgpu (Linux) and gfx110X (Windows); TensileLite Python unit and characterization suites; HOST_ASAN build and quick test on gfx90a; TensileLite coverage floor and ratchet; `pre-commit` |
| **Informational** | HOST_ASAN on gfx942 (opt-in via the `ci:asan` label, explicitly non-blocking); the TensileLite characterization-versus-unit coverage summary card; codecov reports |
| **Unstable / flaky** | Not tracked as a category. hipBLASLt has no `UNSTABLE` tag |

The third row is a real gap and not a claim that nothing is flaky. What hipBLASLt has instead is a
quarantine list, described next.

### Flaky Test Policy

hipBLASLt does not currently tag flaky tests. What it has is
[`clients/tests/data/known_bugs.yaml`](clients/tests/data/known_bugs.yaml), a quarantine list that
matches test cases by their parameters and optionally by GPU architecture
(`known_bug_platforms`). Quarantined cases are excluded from every tier.

The convention in that file is better than the absence of a tag suggests: entries carry a name, a
tracking ticket in a comment, an explanation of the failure mode, and an explicit instruction to
remove the entry once the underlying fix lands so the case re-activates as an enforced gate. That is
close to what the template asks for.

What is missing against the template's expectations:

- No distinction between a *flaky* test (nondeterministic) and a *known-failing* test
  (deterministically wrong, pending a fix). Today both go in the same file.
- No owner field. Tickets are referenced in comments but no person is named in the repository.
- No expiry. Nothing prompts a review of a quarantine entry that has been sitting for a year.
- No automated report of what is currently quarantined and for how long.

A flaky test is not an accepted final state, and neither is an aging quarantine entry.

### Coverage

Code coverage (how much code the tests execute) and test coverage (how much of the intended
functionality and scenarios are tested) are different things, and hipBLASLt is a good illustration
of the difference. TensileLite could reach high line coverage while leaving whole categories of
kernel-generation behavior unexercised, and the client suite could exercise every supported
datatype while touching a small fraction of the C++ library's lines.

**Code coverage as measured today:**

| Scope | Tool | Measured in | Enforced |
| --- | --- | --- | --- |
| TensileLite Python | `coverage.py` via `tox -e coverage-unit` | GitHub Actions, on any change under `projects/hipblaslt/tensilelite/**` | Yes: whole-project floor plus per-file ratchet with 1 pp tolerance |
| TensileLite C++ host library | `tox -e coverage-cpp` | Math CI | Reported to codecov, not enforced |
| hipBLASLt C++ library | Optional `HIPBLASLT_ENABLE_COVERAGE=ON` build | Not run in CI | No |

The GitHub Actions coverage lane is CPU-only and takes roughly seven minutes. It runs the
characterization suite and the pure unit suite once each under coverage, unions the results,
enforces both floors on the combined data, and renders a non-gating summary card attributing
coverage to each suite so the next refactor targets are visible. That lane is deliberately scoped
and is expected to retire once the characterization-to-unit conversion finishes.

**Targets.** TensileLite PRs are expected to reach 80% patch coverage on modified Python, with
essentially full patch coverage required for new Python files and for changes to files already at
full coverage. Files with no meaningful coverage today, notably the assembly kernel writers and
`Solution.py`, are named exceptions where a justification comment is expected instead. The whole
project is expected to reach 80% by end of 2026, from a starting point in the low thirties.

**Scope and exclusions.** Coverage measurement is Python-focused. The kernel writers and assembly
generation modules are outside the characterization scope and are excluded from the per-file floors.
Linux and Windows coverage are not tracked separately; coverage is measured on Linux only.

**No C++ coverage target exists**, for the reasons given in
[Unit Testing Strategy](#unit-testing-strategy). Setting one before the host-side code is linkable
in isolation would produce a number nobody could act on.

### Nightly Validation

Beyond PR validation, the following run on a nightly or postsubmit cadence rather than per PR:

- **Additional hardware.** gfx950 runs hipBLASLt tests on postsubmit and nightly, but not on PRs, due
  to runner capacity ([ROCm/TheRock#3288](https://github.com/ROCm/TheRock/issues/3288)). Additional
  gfx families (gfx90a, gfx103X, gfx110X on Linux, gfx1151, gfx120X) are covered nightly.
- **Wider test tiers.** Nightly runs use the `comprehensive` tier; prerelease runs use `full`.
- **Full ASAN.** TheRock's nightly ASAN lane runs device-side instrumented builds with tests, which
  the per-PR hipBLASLt lane does not.
- **TensileLite GPU tests.** The GPU-marked subset of the TensileLite suite runs on scheduled builds
  rather than per PR.
- **Multi-architecture TensileLite kernel-config suite.** Math CI runs a broad YAML kernel-config
  suite across gfx90a, gfx942, gfx950, and gfx12 on PRs; that architecture breadth is what makes it
  the slowest and most valuable functional signal hipBLASLt has.

### Supported Configurations

Where hipBLASLt tests actually run. This table describes validation of *this component's tests*, not
the set of architectures the library supports or builds for.

| Configuration | Validation level | Frequency | Notes |
| --- | --- | --- | --- |
| Linux, gfx94X-dcgpu | Full | PR (quick), nightly (comprehensive) | Primary PR test target. 6 shards |
| Linux, gfx90a | Partial | PR (HOST_ASAN quick tier), nightly | Sanitizer lane's default architecture |
| Linux, gfx950-dcgpu | Full | Postsubmit and nightly, **not PR** | Runner capacity, ROCm/TheRock#3288 |
| Windows, gfx110X | Partial | PR (quick), nightly | 1 shard |
| Windows, gfx1151 | Partial | Nightly | Forced to the quick tier regardless of requested tier, for memory reasons |
| Linux, gfx90a / gfx942 / gfx950 / gfx12 | Partial (TensileLite kernel-config suite only) | PR, via Math CI | Broadest architecture coverage on a PR |
| Linux, gfx103X / gfx120X | Partial | Nightly | |

**Explicitly not tested**, so that nothing is assumed at release time:

- **Multi-GPU.** There are no multi-GPU tests at all.
- **Windows beyond gfx110X and gfx1151.** No other Windows architecture runs hipBLASLt tests.
- **HMM / managed memory**, except in the `full` tier, which requires a capable host and does not run
  in PR or nightly CI.
- **gfx950 on pull requests.** A change can merge without ever having run hipBLASLt's GTest suite on
  gfx950.
- **Any architecture not listed above.** The library may build for it; its tests do not run.

## Sanitizer Coverage

hipBLASLt has a dedicated per-PR sanitizer lane, which is unusual among ROCm components and is one
of the stronger parts of its story.

**What runs.** The `hipBLASLt ASAN CI` workflow
([`.github/workflows/hipblaslt-asan-ci.yml`](../../.github/workflows/hipblaslt-asan-ci.yml)) triggers
on any pull request touching `projects/hipblaslt/**` or the hipBLASLt test driver. It builds
hipBLASLt through TheRock with `HOST_ASAN` and then runs the quick tier of `hipblaslt-test` under
the sanitizer runtime on real hardware.

| Sanitizer | What it catches | Where it runs | Gating |
| --- | --- | --- | --- |
| **HOST_ASAN** (host-side AddressSanitizer) | Host-side heap and stack overflows, use-after-free, and leaks (via LeakSanitizer) in the library and client code | Every PR touching hipBLASLt, gfx90a | **Yes** |
| **HOST_ASAN** | Same, second architecture | Opt-in via the `ci:asan` PR label, gfx942 | No, explicitly non-blocking |
| **Full ASAN** (host plus device instrumentation) | Adds device-side memory errors in kernels | TheRock nightly and manual dispatch, gfx94X | No |
| **TSAN** | Data races | **Nowhere.** Build options exist; no CI lane uses them | No |

**Runtime configuration.** The lane sets a large ASAN quarantine, a LeakSanitizer suppression file
at `test/therock/lsan.supp` in the repository root, an explicit symbolizer path, and `HSA_XNACK=1`
(required for sanitized ROCm builds). The suppression file is the thing to look at first when a leak
report appears that seems to come from outside hipBLASLt.

**How to build it yourself.** For a standalone hipBLASLt build, `-DHIPBLASLT_ENABLE_ASAN=ON` or
`-DHIPBLASLT_ENABLE_TSAN=ON`, with `-DTENSILELITE_ENABLE_HOST_ASAN=ON` for the TensileLite host
library. Within a TheRock superbuild, `-DhipBLASLt_SANITIZER=HOST_ASAN` instead; hipBLASLt's own
sanitizer options intentionally stand down when the superbuild is driving.

**GPU-specific limitations.** Host ASAN does not instrument device code, so nothing in the generated
kernels is checked by the gating lane. Device-side ASAN requires XNACK-capable configurations, is
substantially slower, and is why full ASAN is nightly rather than per PR.

**Explicitly not covered:** thread safety (no TSAN lane, despite the build option existing);
undefined behavior (no UBSAN); device-side memory errors on any per-PR lane; and any test outside
the quick tier, since the sanitized run uses the quick tier only.

## Why We Test This Way

The strategy above is not the one you would design from scratch. It is the one the architecture
allows, and it is worth explaining why, so that proposals to change it start from the real
constraints.

**The product is generated, so the generator is where unit testing pays.** Because kernels come out
of TensileLite rather than out of hand-written C++, a bug in kernel selection or code generation
affects every consumer of the affected shape. That code is pure Python producing text, which makes
it cheap to test exhaustively and cheap to characterize. Investing there buys more per hour than
anywhere else in the component, which is why the deepest test infrastructure in hipBLASLt sits in
`tensilelite/` and not in `library/`.

**Correctness of a GEMM is not decidable without hardware.** A kernel can be structurally valid,
compile cleanly, and produce wrong numbers on one architecture at one tile size. No amount of
host-side testing detects that. This forces a large integration suite and sets a floor on how fast
the signal can possibly be.

**The combinatorial space is the real adversary.** Datatypes, transposes, epilogues, scaling
formats, batching modes, and shapes multiply out to a space nobody can test exhaustively. The suite
is therefore a sampling strategy, and the interesting engineering question is always which samples
earn their runtime. That is why the tiers exist and why pruning redundant numerical variants is a
live concern rather than a settled one.

**The characterization suite exists because of a specific risk.** TensileLite is on a path to absorb
Tensile and take on a long optimization roadmap. Large refactors of unfamiliar code without a
behavioral net produce silent regressions found downstream, months later, by someone else. The 99
goldens are the net. They deliberately encode current behavior including its bugs, because a net
that only holds correct behavior does not catch a refactor that changes incorrect behavior into
differently incorrect behavior.

**Quarantine entries encode real incidents.** The entries in `known_bugs.yaml` are not test debt in
the abstract; each is a concrete defect with a ticket, several of them numerical (TF32 with infinite
inputs returning NaN, a bf16 output-store address overflow producing silent wrong results at large
leading dimensions). The bf16 overflow entry is instructive: a bit-exact reproducer was landed
*before* the fix, quarantined, with instructions to remove the quarantine once the fix lands so the
reproducer becomes an enforced gate. That is the pattern worth generalizing.

**The current balance is defensible but lopsided.** Integration testing carries almost all of the
correctness load; unit testing carries the generator; performance testing carries nothing
automatically. The first two are appropriate. The third is a genuine weakness for a library whose
entire reason for existing is throughput.

## Key Quality Concerns

The things that must not break, in rough order of how much damage a miss does.

**1. Numerical correctness.** A GEMM that returns wrong numbers, silently, is the worst failure this
library can produce, because it propagates into model outputs with no error anywhere. It is
validated by the client suite comparing against a CPU reference with tolerance checks across the
datatype and shape space, plus dedicated ULP-level tests. Silent-wrong-answer bugs are exactly what
the reproducer-then-quarantine pattern in `known_bugs.yaml` is designed to prevent from recurring.

**2. Performance.** Users choose hipBLASLt for throughput; a correctness-preserving change that costs
20% is a real defect. This is currently validated by hand, or not at all. It is the largest gap in
this document.

**3. Kernel selection and generation correctness.** A regression in TensileLite affects every
consumer at once and can be invisible in a small test set. Validated by the Python unit suite, the
characterization goldens, and the multi-architecture kernel-config suite in Math CI.

**4. Memory safety.** Wrong-sized workspaces, out-of-bounds host buffers, and leaks in long-running
inference processes. Validated by the per-PR HOST_ASAN gate on gfx90a and by nightly full ASAN.
Device-side memory errors are covered only nightly.

**5. API and ABI compatibility.** hipBLASLt is consumed by frameworks; a breaking change is felt
immediately and widely. Validated by the auxiliary and descriptor API tests and by downstream
integration outside this repository.

**6. Packaging and installed-tree layout.** The runtime locates its kernel libraries by path with no
fallback, so a packaging change can break the library at first dispatch while every unit test still
passes. Validated by the pre-flight layout check that runs ahead of the GTest binary in CI.

## Release Validation

What is expected before a release is considered validated. Parts of this are process that lives
outside this repository, so this section describes the component's contribution to it rather than
the whole release gate.

- **Release-branch validation.** The `comprehensive` tier across the supported architecture set,
  rather than the `quick` tier that PRs run.
- **Supported GPU and OS validation.** Explicit runs on the architectures being claimed for the
  release, including the ones that do not run in PR CI. gfx950 in particular must be validated
  deliberately at release time, because no PR has ever run the client suite on it.
- **`full` tier and HMM.** The managed-memory cases run only here, on a capable host.
- **Known-gap review.** The quarantine list in `known_bugs.yaml` should be walked before sign-off:
  every entry is a known defect shipping in the release, and each should be a conscious decision
  rather than an accumulated default.
- **Performance review.** Currently manual and informal. This is the release step most in need of
  automation.
- **QA handoff.** System-level and release validation are executed by QA outside this repository.

## Dependencies and Validation Handoffs

Where confidence comes from and where ownership changes hands.

| Dependency | Owning team | How it is validated | Known gap |
| --- | --- | --- | --- |
| TheRock (build and shared test infrastructure) | TheRock team | PR, nightly, and release lanes | Runner capacity limits which architectures run per PR |
| Math CI (multi-architecture TensileLite suite) | DevOps / Math CI | Per-PR checks on the same PR | Configuration is not visible from this repository, so contributors cannot see what it runs or whether it is required |
| HIP, ROCr, compiler toolchain | Core ROCm teams | Consumed via TheRock; validated by their own CI | A toolchain regression surfaces here as a hipBLASLt test failure, and triage cost lands on this team |
| `rocisa` | Component team (in this repository) | Own tox environment and CI lane | |
| Downstream frameworks | Framework teams | Integration testing outside this repository | No pre-merge signal; regressions are found after the fact |
| QA validation | QA | Release qualification | |

The honest summary is that hipBLASLt depends on two CI systems it does not own, and the boundary
between them is not documented anywhere a contributor can see. That is the most common source of
"I thought CI covered that" confusion in this component.

## Coverage Expectations by Change Type

| Change type | Expected validation |
| --- | --- |
| New hardware-independent logic (TensileLite Python) | Unit test, with patch coverage at the stated target |
| New hardware-independent logic (C++) | Unit test if the code is reachable from `hipblaslt-test`; otherwise say so in the PR and prefer a structure that is reachable |
| New on-device behavior | Client test case with numerical validation against the CPU reference |
| New public API | API or descriptor test in the auxiliary suite |
| Bug fix | A regression test that fails without the fix. If the fix is not yet available, land the reproducer quarantined in `known_bugs.yaml` with a removal note, following the existing pattern |
| Change to TensileLite behavior | Update the affected golden on the smallest node id, and explain the diff in the PR description |
| Performance-sensitive path | Before-and-after benchmark numbers in the PR description, since no automated comparison exists |
| New GPU or OS support | Validation on that configuration, plus an update to the supported-configuration table above |
| Packaging or build change | Install and layout validation; the pre-flight layout check must pass |

## Improvement Roadmap

Ordered by value per unit of effort, not by ambition.

**Near term, cheap and unblocking:**

1. **Fix the tier filter in the TheRock test driver** so tiers above `quick` select their categories
   instead of running the whole binary. One-line fix; makes the documented taxonomy real.
2. **Make the installed-artifact lane's snapshot behavior deliberate.** Either ship syrupy with the
   installed test tree so the goldens are checked there too, or state in the lane that snapshot
   coverage is intentionally left to the source lanes. Today it is a silent skip that reads like an
   accident.
3. **Publish the required-check list.** Document which checks actually block a merge, so contributors
   and reviewers stop guessing.
4. **Split `known_bugs.yaml` semantics**: distinguish flaky from known-failing, and add an owner and
   a review date to each entry.

**Medium term, the structural unlock:**

1. **Create a host-only link target for library internals**, resolving the include-path collision, so
   C++ unit tests can reach validation, enum and string mapping, tuning-override parsing, and
   workspace-sizing logic without linking the whole shared library. Everything in the C++ unit-test
   backlog is blocked behind this one change.
2. **Unify the three parallel enum-to-string tables** (client, library, and test-data YAML), or at
   minimum add a test that asserts they agree. They can drift today, and a drift produces confusing
   test-selection behavior rather than an obvious failure.
3. **Extract validation ahead of dispatch** so argument-error paths are reachable without a GPU.

**Longer term, the real gap:**

1. **Automated performance regression detection.** Stored per-architecture baselines, a defined
   threshold, and an automated comparison at some cadence. Nightly first; PR-level gating only if it
   can be made fast and stable enough not to become the thing everyone reruns.
2. **Graduate mutation testing** from a report-only pilot to a maintained signal on the modules where
   it has demonstrated value.
3. **Prune redundant numerical variants** in the client suite to buy back runtime, and spend it on
   architecture breadth per PR.

## Known Risks and Gaps

Stated plainly, so none of these are a surprise at release time.

| Gap | Regression risk | Impact | Mitigation today | Owner |
| --- | --- | --- | --- | --- |
| No automated performance regression detection at any cadence | High | High | Manual before-and-after benchmarking by whoever suspects an impact | TBD |
| gfx950 never runs the client suite on a PR | High | High | Postsubmit and nightly coverage; deliberate validation at release | TBD |
| The installed-artifact test lane silently skips the snapshot tests, because syrupy is not part of the installed tree's requirements | Low | Low | The goldens are already enforced upstream in the source lanes; the skip is stated in `conftest.py` rather than hidden, but it reads like an accident to anyone scanning the run | TBD |
| Tiers above `quick` apply no filter in the TheRock lane, so the taxonomy is only half-real | Medium | Medium | CTest honors the tiers correctly when used | TBD |
| Very little of the C++ library is unit-testable; the blockers are structural | Medium | Medium | Heavy integration coverage compensates for correctness, at the cost of slow feedback | TBD |
| No flaky-test tagging, owner, or expiry convention | Medium | Medium | `known_bugs.yaml` quarantine with ticket references and removal notes | TBD |
| Which checks are actually required to merge is undocumented | Medium | Medium | Institutional knowledge | TBD |
| TSAN build options exist but no CI lane uses them; no UBSAN at all | Low | High if hit | None. Thread-safety bugs would be found downstream | TBD |
| No multi-GPU tests | Low | High if hit | None in this repository | TBD |
| Three parallel enum-to-string tables can drift | Low | Medium | None automated | TBD |
| Mutation testing is a report-only pilot on a small slice | Low | Low | It is a signal, not a gate, and is treated as such | TBD |
| Submodule-bump pull requests run a reduced test set relative to source changes | Medium | Medium | Owned outside this component; noted here because failures have been merged past | TBD |

Owners are marked TBD rather than assigned unilaterally. Filling this column in is part of taking
this document out of draft.

## Owners and Review Cadence

**Document owner:** Tony Davis (@tony-davis), responsible for keeping this document accurate.

**Test ownership** is currently distributed rather than assigned: the client GTest suite, the
TensileLite Python suites, the characterization goldens, and the CI lanes each have people who work
on them, but no named owner recorded here. Recording named owners is a prerequisite for the flaky
test policy to mean anything, since "every flaky test has an owner" requires there to be owners.

**Review this document when:**

- a major architectural change lands, particularly anything that changes what is testable without a
  GPU;
- a new test pattern or test lane is introduced, or an existing lane changes what it gates;
- a significant regression escapes to a consumer, since that is direct evidence about what this
  strategy is missing;
- before a major release, alongside the known-gap review;
- when release validation assumptions change.

At minimum, revisit the [Known Risks and Gaps](#known-risks-and-gaps) table quarterly. The measure
of whether this document is working is whether that table shrinks.

## For New Contributors

When adding or modifying functionality:

1. Understand which of the [Key Quality Concerns](#key-quality-concerns) your change touches.
2. Add the appropriate level of validation (see below).
3. For a bug fix, add a regression test that fails without the fix.
4. Make sure the required CI gates pass, and do not assume a green check means your change was
   tested on the architecture you care about. Check the
   [supported configurations](#supported-configurations) table.
5. Update this document if the testing strategy changes.

### Choosing the Right Test Type

- **Can the behavior be validated without GPU hardware?**
  - Yes, and it is in TensileLite Python: add a unit test in `tensilelite/Tensile/Tests/unit/`.
  - Yes, and it is C++: add a host-only case in `clients/tests/src/`, if the code is reachable. If it
    is not reachable, say so in your PR rather than reaching for a GPU test by default; the
    reachability problem is a known gap and evidence of it is useful.
  - No: add an integration test case in the client YAML data.
- **Are you changing TensileLite behavior?** Expect a golden diff. Update the smallest node id, read
  every changed line, and explain it in the PR description. Never bulk-regenerate.
- **Is this a public API change?** Add an API or descriptor test in the auxiliary suite.
- **Is this a bug fix?** Add a regression test that fails without the fix.
- **Is this performance-sensitive?** Run the benchmarks yourself and put the numbers in the PR. No
  automation will do it for you.
- **Does this add GPU or OS support?** Validate on that configuration and update the
  supported-configuration table.

## How This Document Is Used

This is a living strategy artifact, not a one-time deliverable. It is intended for regression
analysis (what was supposed to catch this?), quality and release-readiness reviews, engineering
onboarding, and planning CI improvements.

The most useful thing it can do is make the sentence "we assumed CI was covering it" less common.
Every gap above is a place where that sentence would otherwise get said, later, with worse
consequences.
