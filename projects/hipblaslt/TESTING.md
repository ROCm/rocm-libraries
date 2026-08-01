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

This is where most of hipBLASLt's hardware-independent testing lives. The code generator is pure
Python producing text, so essentially all of it is testable with no GPU. How much of that testing is
*unit* testing, as opposed to characterization scaffolding, is a separate question that this section
returns to below. It matters more than the headline coverage number does.

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

**Characterization coverage is not unit coverage, and the coverage numbers do not distinguish them.**
This is the most important caveat on every percentage in this document. A characterization golden
proves that behavior did not change. It does not prove the behavior is correct, and by design some
goldens pin behavior that is known to be wrong. A unit test asserts what the code *should* do. So a
line reached only by a characterization test is protected against accidental change but unverified,
and it still owes a real test.

The scaffolding is deliberate and temporary. It exists so the consolidation refactor can proceed
safely, and the intent is for unit tests to replace it as the code becomes unit-testable. The
difficulty is that the enforced floors cannot see that migration at all. Coverage is measured on the
union of the two suites, so converting a line from characterization protection to a genuine unit test
leaves every enforced number identical. A file can sit at 80% while almost all of that 80% is
scaffolding, and no gate will say a word.

What does show it is the characterization-versus-unit summary card, rendered into the GitHub Actions
run summary by the coverage lane. It splits every measurable statement into four buckets that sum to
100%: reached by both suites, by characterization only, by unit only, and by no test at all. The
characterization-only count is the migration debt, and the goal is for it to fall toward zero. The
card also ranks the largest files by statement count with each file's unit-suite and
characterization-suite percentages side by side, which is how the next refactor target gets picked: a
high characterization percentage next to a low unit percentage is a file still leaning on
scaffolding. The two suites are kept disjoint in the coverage lane specifically so this attribution
means something.

Worth being blunt about the consequence: the one number that measures real progress is the one number
nothing enforces. Reading a file's coverage without reading its split is how a team convinces itself
it is further along than it is.

The discipline that goes with goldens is documented in the suite's
[README.md](tensilelite/Tensile/Tests/unit/characterization/README.md) and is worth stating here
because it is easy to get wrong: **never run a blanket `pytest --snapshot-update`.** It rewrites
every golden at once and produces a green run that proves nothing. Update the smallest node id you
intend to change, read the resulting diff, and explain the behavior change in your PR description.

**The goldens are enforced.** Any pull request touching `projects/hipblaslt/tensilelite/**` runs the
`Component CI: TensileLite coverage` GitHub Actions lane, which executes `tox -e coverage-unit` over
the characterization tree with syrupy installed. A stale golden fails that check. The same tox
environment also runs in Math CI's `tensilelite-unit-codecov` job, so two independent lanes assert the
same snapshots, though only the GitHub Actions one is required to merge.

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

**Mutation testing** is how the scaffolding earns its keep. Coverage only says a line was executed;
a surviving mutant says the suite did not notice when that line's behavior changed, which is exactly
the failure mode a golden-based suite is prone to. It is the only signal available today that
distinguishes a characterization test that would catch a regression from one that merely runs the
code, so widening it is what makes the scaffolding trustworthy while the unit tests are still being
written. Today it is a report-only pilot on a five-file slice, run through `tox -e mutation-unit` and
configured in `pyproject.toml`. It is not a gate and does not run in CI. Accepted equivalent mutants
and every `# pragma: no mutate` are justified in `DECISIONS.md`. A series of PRs widening the
mutation-hardened surface starts at
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

**Coverage expectation.** The enforced whole-project floor for TensileLite Python is `fail_under = 75`
in [`tensilelite/pyproject.toml`](tensilelite/pyproject.toml), which is deliberately the only place
that number lives. The comment beside it records the intent: the GPU-less `coverage-unit` run
measures around 79%, 80% is the target, and the floor is set below the measured value on purpose so
that ordinary run-to-run noise cannot trip an exact cutoff. Per-file floors ratchet separately.

Read that 79% with the caveat above firmly in mind: it is union coverage across the unit and
characterization suites, so it describes how much code is *protected*, not how much is *verified*.
There is no enforced target on the unit-only share, which is the number that actually tracks the
migration. For the C++ side there is no coverage target at all today, because with the current
structure a target would be aspirational rather than actionable.

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

### Build-Time Validation of Library Logic

This one is easy to miss, because it does not look like a test and does not report like one. It is
nonetheless a real gate, and it covers a class of defect nothing else in the component touches.

hipBLASLt's kernel selection is driven by library logic YAML: generated, tuned data describing which
solution serves which problem shape on which architecture. That data can be wrong in ways no code
review catches, and a bad entry surfaces at runtime as a kernel-selection failure on one shape on one
architecture, which is an expensive way to find out.

`TensileLogic --check-all` validates it before any of it is used. It checks chip IDs, matrix
instructions, work-group shapes, the XCC work-group mapping, and custom kernel declarations across
every logic file. It reads YAML only, so it needs no GPU and no compiled kernels, and it is fast.

**Where it runs is the odd part.** It is wired into the build, as a CMake custom command in
[`cmake/HipBLASLtCodegen.cmake`](cmake/HipBLASLtCodegen.cmake) that runs ahead of
`TensileCreateLibrary` and writes a stamp file. So it executes on every build that generates kernels,
including every developer's local build, and a failure stops the build. That gives it excellent
reach and costs three things: it has no check name in the pull request, it produces no test report,
and it cannot be run in isolation by CI. It also cannot be tightened in place, because any stricter
setting would fail local developer builds as readily as CI ones.

Its known-bug list,
[`tensilelite/Tensile/TensileLogic/known_bugs.yaml`](tensilelite/Tensile/TensileLogic/known_bugs.yaml),
is the best-structured quarantine in the component (see
[Known Bugs and Expected Failures](#known-bugs-and-expected-failures)). Entries are keyed on the
logic file path plus the solution's `SolutionNameMin`, a content-derived name chosen so that keys
survive library re-tuning instead of shifting with a positional index, and each entry carries a
`ticket:` field. The checker re-validates every entry and reports the ones that no longer reproduce,
so a fixed bug is detected rather than skipped forever. All 14 current entries document the same
gfx950 validation drift under ROCM-7144.

`--strict-known-bugs` turns that detection into a failure, but it defaults off and nothing passes it
today, so a stale entry only warns. Enforcing it is tracked in AIHPBLAS-4196, which proposes the
right fix: a dedicated GitHub Actions job running
[`scripts/run_tensile_logic_check.py`](scripts/run_tensile_logic_check.py) with the flag, rather than
tightening the in-build command. That ticket also names a remaining hole, which is that an orphaned
entry whose `solution_name` resolves to nothing currently matches nothing and is silently ignored, so
strict mode is not yet a complete dead-entry detector. Extending the gate to cover derived-parameter
assignment is tracked in AIHPBLAS-3575.

The check arrived in [PR #5039](https://github.com/ROCm/rocm-libraries/pull/5039) and was re-keyed
onto solution names in [PR #9355](https://github.com/ROCm/rocm-libraries/pull/9355).

### Performance and Benchmarking Testing

hipBLASLt is a performance library, so this section deserves to be read carefully. The distinction
that matters: measurement and comparison are automated, but the verdict is not. Nothing decides that
a number is bad and fails the build.

| Item | Detail |
| --- | --- |
| Stack layer | Core SDK (math library) |
| Metrics measured | GFLOP/s, memory bandwidth (GB/s), latency (µs). Optionally clock frequencies, achieved efficiency, and memory read/write bytes |
| How benchmarks are run | `hipblaslt-bench` for a single problem or a YAML batch; `hipblaslt-perf` to run named suites and aggregate CSV over repeated samples; `hipblaslt-cotenant` to measure under CU contention |
| Automated runs | Two Math CI jobs, `perfci` and `performance` (described below). Neither is a merge gate |
| Reference for comparison | A second build of the merge target (`develop`) produced in the same job, benchmarked on the same machine and tagged `build_reference` against the change's `build_new` |
| Where results are stored | Ingested into rocPTS, AMD's performance tracking service, and viewable from a Kibana dashboard and the PTS web app. Local runs write CSV to the developer's workspace |
| Regression threshold | **None defined.** The ingest step uploads both datasets and fails only if the upload itself fails. The tuning flow's uplift threshold is a candidate filter, not a regression gate |
| Gating approach | Human review of the dashboard |

**The two automated jobs.** `perfci` runs on pull requests. It builds the change and a `develop`
reference, benchmarks both with the `ci_perf_job` suite on gfx950 (specifically an MI350X, pinned by
device ID because the job requires it), uploads both datasets to rocPTS, and posts a comment on the
pull request linking the run. It is configured not to report a check to GitHub, so it appears as a
comment rather than as a check row. `performance` is the broader job: the `all` suite on gfx942 and
gfx950, throttled to one run per day, reporting a check.

Note that the `ci_perf_job` suite defined in this repository is invoked by Math CI, not from anything
in this tree. Searching this repository for callers finds none, which is misleading.

**Gating, stated honestly:**

| Gating level | Status | Notes |
| --- | --- | --- |
| PR-level automated measurement | **Yes**, via `perfci` | Change and reference are benchmarked side by side and both land in rocPTS |
| PR-level automated gate | **No** | No threshold is defined and the job is not in the gating set. A regression shows up as a number on a dashboard that someone has to look at |
| Daily automated comparison | **Yes**, via `performance` | Same shape, wider suite, two architectures, once per day |
| Automated regression alerting | **No** | Nothing watches the dashboard and tells anyone. This is the real gap, not the absence of measurement |
| Release qualification | **No documented gate** | Performance is discussed at release time but there is no in-repo criterion |

Related tooling that is sometimes mistaken for regression testing: `utilities/geko/` is a GEMM
kernel optimizer that searches for better kernels and benchmarks candidates during tuning, and
`utilities/QuickTune/` performs offline per-workload tuning from captured logs. Both produce
performance numbers as part of tuning rather than as a regression signal. Similarly, TensileLite's
benchmark and library-logic phases generate performance data used to *select* kernels.

**Known gaps:** no defined threshold, no automated alerting on a regression, no gate at any cadence,
and no gate on library size, kernel count, or build time. Architecture coverage is also narrower than
the library ships for: `perfci` measures gfx950 only. A performance regression is caught today by a
human reading a dashboard, or by a downstream consumer.

## Pre-submit / CI Gates

hipBLASLt is validated by two CI systems, which is the single most important thing to understand
about its signal. GitHub Actions runs the build and the shared TheRock-based test lanes. A separate
internal Jenkins-based system ("Math CI") builds and tests on AMD-hosted GPU runners and posts checks
named `mci/rocm-libraries/...` onto the same pull request. Both appear in the PR checks list.

Math CI's configuration lives in an AMD-internal repository rather than in this tree, so a
contributor reading this repository cannot see what it runs. For hipBLASLt it defines roughly a dozen
job types, of which exactly three are configured as gating:

- **`precheckin`** builds hipBLASLt from source and runs its client test suite on gfx90a, gfx942,
  gfx950, and gfx12, plus a compile-only gfx1250 configuration. This is the broadest hardware
  coverage hipBLASLt gets on a pull request, and it is the only per-PR signal on gfx950.
- **`static-analysis`** runs a static analysis scan over the source.
- **`preliminary`** runs the TensileLite `common`-marked suite under tox on gfx12, gfx90a, gfx942,
  and gfx950. It first diffs the change against `develop` and skips entirely when nothing under
  `tensilelite/`, `shared/stinkytofu/`, or `shared/origami/` has changed, so on a pull request that
  touches neither, it passes without running anything.

Other Math CI jobs post checks without gating. The one worth knowing is
`tensilelite-unit-codecov`, which runs the TensileLite Python and C++ coverage environments on gfx950
and uploads to codecov under the `TensileLite-Unit` and `TensileLite-CPP` flags. It reports a check,
but that check is not required.

### Validation Gates and Ownership

| Validation area | Required before merge | Owner | Responsibility |
| --- | --- | --- | --- |
| Build (Linux and Windows) | Yes | CI / DevOps + TheRock | Maintain jobs, runners, and pipeline health |
| Library logic validation (`TensileLogic --check-all`) | Yes, implicitly | Component team | Runs inside the build ahead of codegen, so it blocks any build that generates kernels. It has no separate check name and produces no test report |
| Unit tests (TensileLite Python) | Yes | Component team | Create, maintain, review |
| Integration / smoke tests (client GTest) | Yes | Component team | Validate behavior across key scenarios |
| Characterization goldens | Yes, when `tensilelite/` is touched | Component team | Review every golden diff; never bulk-regenerate |
| HOST_ASAN build and quick test | Yes, on gfx90a | Component team | Keep the sanitizer lane green |
| Code coverage floor and ratchet (TensileLite) | Yes, when `tensilelite/` is touched | Component team / CI | Floors move up only |
| Formatting and lint (`pre-commit`) | Yes | CI / DevOps | Maintain hooks |
| Static analysis | Yes | CI / DevOps | Gating Math CI job (`static-analysis`) |
| Client test suite on gfx90a / gfx942 / gfx950 / gfx12 | Yes | Component team | Gating Math CI job (`precheckin`) |
| Performance | No | Component team | Measured on every PR by `perfci`, but nothing gates on the result (see above) |
| Shared validation infrastructure | N/A | TheRock team | Provide shared build and test infrastructure |
| System validation | N/A | QA | Execute system-level and release validation |
| Release qualification | N/A | Component team + QA + TPM | Confirm readiness, review known risks |

A caveat on this table: the Math CI rows are taken from that system's gating configuration, which is
authoritative but lives outside this repository. Which GitHub Actions checks are configured as
*required* in branch protection is not documented anywhere a contributor can see, and the answer is
not consistently understood even among the people working on the tests. Publishing the actual
required-check list, in this repository, is tracked as a gap.

### PR Test Classification

| Status | Applies to |
| --- | --- |
| **Trusted gate** | Build (both platforms), which includes the `TensileLogic --check-all` library logic validation; client GTest quick tier on gfx94X-dcgpu (Linux) and gfx110X (Windows); the Math CI client suite on gfx90a, gfx942, gfx950 and gfx12; Math CI static analysis; TensileLite Python unit and characterization suites; HOST_ASAN build and quick test on gfx90a; TensileLite coverage floor and ratchet; `pre-commit` |
| **Informational** | HOST_ASAN on gfx942 (opt-in via the `ci:asan` label, explicitly non-blocking); the TensileLite characterization-versus-unit coverage summary card; the `tensilelite-unit-codecov` check and codecov reports; the `perfci` benchmark comparison |
| **Unstable / flaky** | Not tracked as a category. hipBLASLt has no `UNSTABLE` tag |

The third row is a real gap and not a claim that nothing is flaky. What hipBLASLt has instead is a
quarantine list, described next.

### Flaky Test Policy

hipBLASLt does not currently tag flaky tests, and has no `UNSTABLE` category. A nondeterministic
test today goes into the same quarantine list as a deterministically broken one,
[`clients/tests/data/known_bugs.yaml`](clients/tests/data/known_bugs.yaml), which is described with
the other quarantine mechanisms in
[Known Bugs and Expected Failures](#known-bugs-and-expected-failures) below.

That conflation is the policy gap. A flaky test and a known-failing test need different responses: a
flaky test is a defect in the test, usually fixable by the component team, while a known-failing test
is a defect in the product waiting on someone else. Filing them in one undifferentiated list means
neither gets the follow-up it needs, and no report exists of what is currently suppressed or for how
long.

A flaky test is not an accepted final state, and neither is an aging quarantine entry.

### Known Bugs and Expected Failures

hipBLASLt suppresses or records known-bad behavior in seven different places. They accumulated
independently, they use different formats, and they are not governed as one thing. Anyone reasoning
about "what do we currently know is broken" has to check all seven.

| Mechanism | What it suppresses | Ticket linkage | Detects its own fix? |
| --- | --- | --- | --- |
| [`clients/tests/data/known_bugs.yaml`](clients/tests/data/known_bugs.yaml) | Client GTest cases matched by parameters, optionally per architecture via `known_bug_platforms`. Excluded from every tier | Comment convention | **No.** The case never runs, so nothing can observe a fix |
| `GTEST_SKIP()` in client sources | Individual cases at runtime | None | Not applicable, and mostly not bugs: these are environment guards (no GPU present, no Stream-K kernel selected for the problem) |
| [`tensilelite/Tensile/TensileLogic/known_bugs.yaml`](tensilelite/Tensile/TensileLogic/known_bugs.yaml) | Library-logic validation failures, keyed on the logic file path plus the solution's `SolutionNameMin` | Structured `ticket:` field | **Partly.** The check re-validates each entry and reports stale ones, but only warns by default |
| Filename-driven marks in `Tensile/Tests/common/config_helpers.py` | Any config YAML whose path contains `xfail`, `wip` or `disabled` | None; the reason lives in a filename | **No**, and non-strict, so an expected failure that starts passing is silent |
| `skip-<arch>` marks in config YAML `TestParameters` | A config on named architectures | Free-text comment | Not applicable |
| Explicit `pytest.mark.xfail` markers | Specific assertions in a Python test | Ticket in the `reason` string | **Yes**, when written `strict=True` |
| Characterization goldens that pin known-wrong behavior | Nothing. The wrong behavior is recorded rather than hidden | ADR under `adr/` with a defect link, required by the reviewer checklist | Not applicable: a fix shows up as a golden diff needing review |

**The distinction that matters is the last column.** Format is cosmetic; what separates a healthy
suppression from rot is whether the mechanism can tell you the underlying bug got fixed. That sorts
the list into three tiers. *Self-cleaning*: the code still runs, and a fix fails the build, forcing
the entry's removal. *Detectable*: something notices, but nothing fails. *Blind*: the code no longer
runs at all, so the entry can outlive its bug indefinitely. The largest surface, the client
quarantine list, is in the blind tier.

**The best-governed example is already in the tree**, and is worth copying rather than redesigning.
The `_ROCM3994_XFAIL` marker in
[`test_amax_true16_activation.py`](tensilelite/Tensile/Tests/unit/test_amax_true16_activation.py)
carries a ticket in its reason, `strict=True` so that a fix turns the unexpected pass into a hard
failure, `raises=AssertionError` so an unrelated crash is not absorbed, a time-box comment naming
when to re-evaluate, and an explicit instruction to delete the marker in the fixing PR. The test
still executes; it is quarantined, not disabled. Everything a governance policy would ask for is in
those few lines.

The two YAML quarantine files each have half of what the other needs. The client list has the better
prose discipline: named entries, a ROCm ticket, a root-cause explanation, and a note to remove the
entry so the case re-activates as an enforced gate. But it is all in comments, so no tool can act on
it, and the excluded case never runs again. The TensileLogic list has the weaker narrative and the
better structure: a real `ticket:` field, keys chosen so they survive library re-tuning, and a check
that re-validates every entry on each run and reports the ones that no longer reproduce.

**Known gaps.** No mechanism records an owner or a review date. Nothing reports what is currently
suppressed across all seven places, or for how long. The filename-driven marks in `config_helpers.py`
are the weakest link by construction, since a path substring cannot carry a ticket, an owner or a
reason, and the resulting mark is non-strict so a fix is invisible; they have no users in the tree
today, which makes now the right time to decide whether to keep the machinery at all. Separately,
`skip-<arch>` marks appear in around 395 config files with free-text justifications, and while most
are genuine capability statements ("not supported by arch"), some read "not supported yet", which is
deferred work with nothing tracking it. Consolidating all of this under one policy is on the
[roadmap](#improvement-roadmap).

### Coverage

Code coverage (how much code the tests execute) and test coverage (how much of the intended
functionality and scenarios are tested) are different things, and hipBLASLt is a good illustration
of the difference. TensileLite could reach high line coverage while leaving whole categories of
kernel-generation behavior unexercised, and the client suite could exercise every supported
datatype while touching a small fraction of the C++ library's lines.

hipBLASLt has a third distinction on top of those two, and it is the one most likely to mislead: for
TensileLite Python, a covered line may be covered by a *unit test* that asserts intended behavior or
by a *characterization golden* that merely pins current behavior, bugs included. Every enforced
number here is the union of the two and cannot tell them apart. See
[Unit Testing Strategy](#unit-testing-strategy) for why that matters; the short version is that
characterization coverage is scaffolding to be repaid, not testing that is finished.

**Code coverage as measured today:**

| Scope | Tool | Measured in | Enforced |
| --- | --- | --- | --- |
| TensileLite Python, unit and characterization combined | `coverage.py` via `tox -e coverage-unit` | GitHub Actions, on any change under `projects/hipblaslt/tensilelite/**` | Yes: whole-project floor plus per-file ratchet with 1 pp tolerance |
| TensileLite Python, unit-only share | Same lane, reported in the split summary card | GitHub Actions | **No.** Informational only, and it is the number that tracks real progress |
| TensileLite Python, mutation score | `tox -e mutation-unit` | Nowhere; run by hand | No. Report-only pilot on five files |
| TensileLite C++ host library | `tox -e coverage-cpp` | Math CI | Reported to codecov, not enforced |
| hipBLASLt C++ library | Optional `HIPBLASLT_ENABLE_COVERAGE=ON` build | Not run in CI | No |

The GitHub Actions coverage lane is CPU-only and takes roughly seven minutes. It runs the
characterization suite and the pure unit suite once each under coverage, keeping the two selections
disjoint so each line can be attributed to one or both, unions the results, enforces both floors on
the combined data, and renders the non-gating split summary card. That lane is deliberately scoped
and is expected to retire once the characterization-to-unit conversion finishes, which makes the
card's characterization-only count a rough progress bar for the lane's own retirement.

**Targets.** Two different mechanisms carry a number, and they are easy to confuse:

- The **enforced floor** is `fail_under = 75` in [`tensilelite/pyproject.toml`](tensilelite/pyproject.toml),
  checked on the combined characterization-plus-unit dataset, alongside the per-file floors in
  `coverage-baseline.json`. This is what actually fails a run.
- The **codecov target** is 80% project coverage per flag, set in the monorepo's
  [`codecov.yml`](../../codecov.yml) for `hipBLASLt`, `TensileLite-Unit` and `TensileLite-CPP` along
  with every other library. No patch-coverage target is configured. Codecov's report is advisory here
  because the job that uploads it is not a required check.

80% is the direction of travel for the enforced floor, and the ratchet is how it gets there.

Neither mechanism sets a target on the unit-only share, and neither one would notice if the
characterization-only count stopped falling. Both numbers can be fully satisfied by a codebase whose
Python is entirely pinned and barely verified. Setting an explicit target on the unit-only share, so
the migration has a gate and not just a dashboard, is on the [roadmap](#improvement-roadmap).

**Scope and exclusions.** Coverage measurement is Python-focused, and measured on Linux only; Linux
and Windows are not tracked separately. The exclusions in
[`tensilelite/pyproject.toml`](tensilelite/pyproject.toml) cover test, build and packaging paths
rather than product modules. The kernel writers are sometimes described as uncovered exceptions, but
they are not: `KernelWriter.py`, `KernelWriterAssembly.py` and `SolutionStructs/Solution.py` all
carry active per-file floors in the seventies. The genuinely uncovered modules are elsewhere,
including `ExperimentalLibrary.py` at zero and much of `Tensile/Components/`.

**No C++ coverage target exists**, for the reasons given in
[Unit Testing Strategy](#unit-testing-strategy). Setting one before the host-side code is linkable
in isolation would produce a number nobody could act on.

### Nightly Validation

Beyond PR validation, the following run on a nightly or postsubmit cadence rather than per PR:

- **Additional hardware in the TheRock lane.** gfx950 runs there on postsubmit and nightly but not on
  pull requests, due to runner capacity
  ([ROCm/TheRock#3288](https://github.com/ROCm/TheRock/issues/3288)). Math CI does cover gfx950 per
  PR, so this is a gap in one lane rather than in the whole gate. Additional gfx families (gfx90a,
  gfx103X, gfx110X on Linux, gfx1151, gfx120X) are covered nightly.
- **Wider test tiers.** Nightly runs use the `comprehensive` tier; prerelease runs use `full`.
- **Full ASAN.** TheRock's nightly ASAN lane runs device-side instrumented builds with tests, which
  the per-PR hipBLASLt lane does not.
- **TensileLite GPU tests.** In the GitHub Actions lanes, the GPU-marked subset of the TensileLite
  suite runs on scheduled builds rather than per PR. Math CI's `preliminary` job runs the
  `common`-marked suite on GPUs per PR, but only when `tensilelite/` and its shared dependencies are
  touched.
- **Broad performance benchmarking.** The Math CI `performance` job runs the full benchmark suite on
  gfx942 and gfx950 once per day, with results in rocPTS. It measures; it does not judge.

### Supported Configurations

Where hipBLASLt tests actually run. This table describes validation of *this component's tests*, not
the set of architectures the library supports or builds for.

| Configuration | Validation level | Frequency | Notes |
| --- | --- | --- | --- |
| Linux, gfx90a / gfx942 / gfx950 / gfx12 | Client test suite | PR, via Math CI `precheckin` | Broadest per-PR hardware coverage, and the only per-PR signal on gfx950. Also compiles for gfx1250 |
| Linux, gfx90a / gfx942 / gfx950 / gfx12 | TensileLite `common` suite | PR, via Math CI `preliminary` | Skipped unless `tensilelite/` or its shared dependencies changed |
| Linux, gfx94X-dcgpu | Full | PR (quick), nightly (comprehensive), via TheRock | Primary GitHub Actions test target. 6 shards |
| Linux, gfx90a | Partial | PR (HOST_ASAN quick tier), nightly | Sanitizer lane's default architecture |
| Linux, gfx950-dcgpu | Full | Postsubmit and nightly in the TheRock lane, **not PR** there | Runner capacity, ROCm/TheRock#3288. Covered per PR by Math CI |
| Windows, gfx110X | Partial | PR (quick), nightly | 1 shard |
| Windows, gfx1151 | Partial | Nightly | Forced to the quick tier regardless of requested tier, for memory reasons |
| Linux, gfx950 (MI350X) | Benchmarks only | PR, via Math CI `perfci` | Non-gating measurement into rocPTS |
| Linux, gfx942 / gfx950 | Benchmarks only | Daily, via Math CI `performance` | Non-gating |
| Linux, gfx103X / gfx120X | Partial | Nightly | |

**Explicitly not tested**, so that nothing is assumed at release time:

- **Multi-GPU.** There are no multi-GPU tests at all.
- **Windows beyond gfx110X and gfx1151.** No other Windows architecture runs hipBLASLt tests.
- **HMM / managed memory**, except in the `full` tier, which requires a capable host and does not run
  in PR or nightly CI.
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

**Characterization came first because the code was not ready for unit tests.** Much of TensileLite
predates any expectation of testability, so writing real unit tests against it means refactoring it,
and refactoring it without a net is how silent regressions happen. Pinning current behavior first,
then refactoring behind the pins, then replacing the pins with unit tests, is the order the situation
forces. The cost of that order is a coverage number that looks better than the underlying quality
warrants, which is why this document keeps separating the two rather than quoting the union and
moving on.

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
20% is a real defect. Every pull request is benchmarked against a `develop` reference and the numbers
land in rocPTS, but nothing compares them to a threshold or raises an alarm, so catching the
regression still depends on someone looking. It is the largest gap in this document.

**3. Kernel selection and generation correctness.** A regression in TensileLite affects every
consumer at once and can be invisible in a small test set. Validated by the Python unit suite, the
characterization goldens, and the multi-architecture `common` suite run by Math CI's gating
`preliminary` job. Note what the goldens do and do not buy here: they will catch an *unintended*
change to generation, which is the common failure, but they cannot tell you the generated result was
right in the first place, since they were recorded from the same code. For that, the module needs
real unit tests, which is the migration described in
[Unit Testing Strategy](#unit-testing-strategy).

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
  release, including the ones that do not run in PR CI at all: the Windows architectures beyond
  gfx110X, gfx103X and gfx120X, and anything newer than the per-PR set.
- **`full` tier and HMM.** The managed-memory cases run only here, on a capable host.
- **Known-gap review.** The quarantine list in `known_bugs.yaml` should be walked before sign-off:
  every entry is a known defect shipping in the release, and each should be a conscious decision
  rather than an accumulated default.
- **Performance review.** The data exists in rocPTS; the review of it is manual and informal. This is
  the release step most in need of automation.
- **QA handoff.** System-level and release validation are executed by QA outside this repository.

## Dependencies and Validation Handoffs

Where confidence comes from and where ownership changes hands.

| Dependency | Owning team | How it is validated | Known gap |
| --- | --- | --- | --- |
| TheRock (build and shared test infrastructure) | TheRock team | PR, nightly, and release lanes | Runner capacity limits which architectures run per PR |
| Math CI (multi-architecture client, TensileLite, coverage and benchmark jobs) | DevOps / Math CI | Per-PR checks on the same PR, plus daily benchmarks | Configuration lives in an AMD-internal repository, so contributors cannot see what runs or which jobs gate without asking |
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
| New hardware-independent logic (TensileLite Python) | A real unit test asserting intended behavior, keeping the whole-project floor and every per-file floor intact. A characterization golden is not a substitute for new code: pinning behavior you just wrote pins whatever you happened to write |
| Refactoring TensileLite behind existing goldens | Goldens unchanged is the pass condition. If the refactor makes a module unit-testable, converting its characterization coverage to unit tests in the same PR is the highest-value thing you can do, and the split card is where to show it |
| New hardware-independent logic (C++) | Unit test if the code is reachable from `hipblaslt-test`; otherwise say so in the PR and prefer a structure that is reachable |
| New on-device behavior | Client test case with numerical validation against the CPU reference |
| New public API | API or descriptor test in the auxiliary suite |
| Bug fix | A regression test that fails without the fix. If the fix is not yet available, land the reproducer quarantined in `known_bugs.yaml` with a removal note, following the existing pattern |
| Change to TensileLite behavior | Update the affected golden on the smallest node id, and explain the diff in the PR description |
| Performance-sensitive path | Read the `perfci` comparison for the PR and summarize it in the description. Nothing fails on a regression, so saying what the numbers showed is the control |
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
4. **Enforce `--strict-known-bugs` in its own lane** (AIHPBLAS-4196). The detection already exists;
   what is missing is a job that fails on a stale entry. A dedicated GitHub Actions job is the right
   home for it, because the flag cannot be turned on inside the build without failing local developer
   builds. Worth extending to orphaned entries, which are silently ignored today.
5. **Govern known-bug entries as one thing.** Seven mechanisms suppress or record known-bad behavior
   and none of them share a convention. The proposal, which needs team agreement before it becomes
   policy, is four rules: every entry names its ticket in a machine-readable field rather than a
   comment; every entry carries an owner and a review date; the suppressed code keeps running wherever
   the mechanism allows, so a fix can be observed rather than assumed; and a fix fails the build so
   the entry deletes itself. Two mechanisms cannot satisfy the third rule today, so adopting this
   means changing them or accepting a stated exception. Start by splitting flaky from known-failing in
   the client quarantine list, which is the largest and blindest surface.

**Medium term, the structural unlock:**

1. **Put a number on the characterization-to-unit migration.** The split summary card already computes
   the characterization-only statement count. Track it over time and set a target on it, so the
   migration has a gate rather than a dashboard nobody is accountable for. Without this, the union
   floors are fully satisfiable by a codebase that is pinned everywhere and verified nowhere, and the
   scaffolding has no expiry date.
2. **Create a host-only link target for library internals**, resolving the include-path collision, so
   C++ unit tests can reach validation, enum and string mapping, tuning-override parsing, and
   workspace-sizing logic without linking the whole shared library. Everything in the C++ unit-test
   backlog is blocked behind this one change.
3. **Unify the three parallel enum-to-string tables** (client, library, and test-data YAML), or at
   minimum add a test that asserts they agree. They can drift today, and a drift produces confusing
   test-selection behavior rather than an obvious failure.
4. **Extract validation ahead of dispatch** so argument-error paths are reachable without a GPU.

**Longer term, the real gap:**

1. **Turn the performance data into a signal.** The measurement and the reference comparison already
   exist and already run per PR. What is missing is a defined regression threshold, an alert when a
   run crosses it, and a named owner who reads the result. Alerting first; PR-level gating only if it
   can be made stable enough not to become the thing everyone reruns.
2. **Graduate mutation testing** from a report-only pilot to a maintained signal on the modules where
   it has demonstrated value. This is the companion to the migration item above: until a module has
   real unit tests, its mutation score is the only evidence that its goldens would actually catch a
   regression rather than just execute the code.
3. **Prune redundant numerical variants** in the client suite to buy back runtime, and spend it on
   architecture breadth per PR.

## Known Risks and Gaps

Stated plainly, so none of these are a surprise at release time.

| Gap | Regression risk | Impact | Mitigation today | Owner |
| --- | --- | --- | --- | --- |
| No threshold, alert, or gate on the performance data, so a regression is only caught if someone reads the dashboard | High | High | Per-PR and daily benchmarks against a `develop` reference, retained in rocPTS | TBD |
| Benchmark coverage is gfx950-only per PR, and gfx942 plus gfx950 daily | Medium | Medium | Correctness coverage is broader; performance risk on other architectures is carried unmeasured | TBD |
| What Math CI runs and which of its jobs gate is not visible from this repository | Medium | Medium | This document, which is a snapshot and will drift from the internal configuration | TBD |
| The installed-artifact test lane silently skips the snapshot tests, because syrupy is not part of the installed tree's requirements | Low | Low | The goldens are already enforced upstream in the source lanes; the skip is stated in `conftest.py` rather than hidden, but it reads like an accident to anyone scanning the run | TBD |
| Tiers above `quick` apply no filter in the TheRock lane, so the taxonomy is only half-real | Medium | Medium | CTest honors the tiers correctly when used | TBD |
| Enforced coverage counts characterization scaffolding the same as unit tests, so the numbers overstate how much TensileLite Python is actually verified | High | Medium | The split summary card reports the characterization-only share, but it gates nothing and no target is set on it | TBD |
| Mutation testing, the only check that a golden would catch a regression rather than just execute the line, covers five files and runs nowhere in CI | Medium | Medium | Manual runs via `tox -e mutation-unit`; widening PRs are in draft | TBD |
| Very little of the C++ library is unit-testable; the blockers are structural | Medium | Medium | Heavy integration coverage compensates for correctness, at the cost of slow feedback | TBD |
| No flaky-test tagging, owner, or expiry convention, and flaky tests share one list with known-failing ones | Medium | Medium | `known_bugs.yaml` quarantine with ticket references and removal notes | TBD |
| Known-bad behavior is suppressed in seven places with no shared convention, and the largest of them excludes the test entirely, so a fixed bug can stay quarantined indefinitely | Medium | Medium | Per-mechanism discipline is good in places and absent in others; nothing reports the total | TBD |
| A stale `TensileLogic` known-bugs entry only warns, because `--strict-known-bugs` defaults off and no job passes it | Low | Low | The checker does re-validate and report stale entries on every build; enforcement is tracked in AIHPBLAS-4196 | TBD |
| The library logic validation gate is invisible as a check: it runs inside the build, so it has no check name, no test report, and cannot be run in isolation by CI | Low | Medium | It runs on every kernel-generating build including local ones, which gives it good reach; a standalone lane is proposed in AIHPBLAS-4196 | TBD |
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
  - Yes, and it is in TensileLite Python: add a unit test in `tensilelite/Tensile/Tests/unit/`. Write
    a test that asserts what the code *should* do. Reach for a characterization golden only when you
    are pinning behavior that already exists so it can be refactored safely, not when you are adding
    behavior.
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
