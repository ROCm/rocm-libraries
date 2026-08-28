# hipBLASLt Testing Strategy

- **Owner:** T.J. Alumbaugh (@talumbau)
- **Technical Lead:** Tony Davis (@tony-davis)
- **Last Updated:** 2026-08-11

> **In a hurry?** Start with [The short version](#the-short-version). From there, jump to
> [Choosing the Right Test Type](#choosing-the-right-test-type) if you are about to write a test, or
> to [Known Risks and Gaps](#known-risks-and-gaps) if you are deciding how much to trust a green
> check.

This document describes how hipBLASLt is tested today, which signals actually gate a merge, and
where the holes are. It follows the ROCm-wide TESTING.md template. It is written as a description of
the current state rather than an aspirational one: several sections record gaps, and a gap that is
written down is one we can argue about and close.

For the mechanics of *how* the client test flow works (YAML data files, `gentest` expansion, GTest
registration), see the test documentation landing in
[PR #10205](https://github.com/ROCm/rocm-libraries/pull/10205). This document is the strategy layer
and does not duplicate it.

One deliberate habit, since it is unusual for a document of this kind. Where a check exists because of
a specific failure, that failure is written up under a subsection headed
**How this one was learned**. This is not decoration. A control whose motivating failure is recorded
nowhere is indistinguishable from overhead, and overhead gets skipped, left unwired, or removed by
whoever touches it next.
[Build-Time Validation of Library Logic](#build-time-validation-of-library-logic) is a case where
exactly that happened to a validator already sitting in this repository, at a cost of about three
months.

Whoever touches it next is increasingly a coding agent rather than a person, which is the second
reason for writing these up. The narrative sections are the most human part of this document and also
the most useful part of it to a machine: they carry intent, they attach a cost to that intent, and
they leave a trail back to the pull requests and tickets where the reasoning actually happened. An
agent that can see why a guardrail exists will keep it, and will pick up the evidence already
gathered rather than re-deriving a conclusion from scratch or deleting something it reads as dead
weight.

Each write-up sits in a highlighted box, states its finding in one bold sentence, and folds the
narrative behind a collapsed **The full account** toggle, so the finding is unavoidable and the story
stays optional.

## The short version

**Where the confidence comes from.** Overwhelmingly from large GTest suites running real GEMMs on
real hardware and comparing against a CPU reference. The one strong hardware-independent suite is in
a place most people do not look first: TensileLite, the Python code generator that emits the kernels.

**What gates a merge.** The build on both platforms, which quietly includes a validation pass over
the library logic YAML; the client GTest suite on gfx90a, gfx942, gfx950 and gfx12; the TensileLite
Python unit and characterization suites; a host-side AddressSanitizer build and quick test on gfx90a;
and `pre-commit`.

**What does not gate, despite appearances.** The coverage floors fail their own lane, but that lane
is not a required check. Pull requests are not benchmarked, so there is no performance signal on a
change to gate on. The gating job named `static-analysis` is a sensitive-word
scan for disclosure rather than a code analyzer, and no static analysis runs on the C++ library on a
pull request from any source.

**The biggest gap is performance.** hipBLASLt exists for throughput, and pull requests are not
benchmarked. What does exist is a nightly lane outside this repository that measures a build of the
whole monorepo, so a regression surfaces a day later at the earliest, against a range of commits,
and reaches the lane's owners rather than the change's author. Second is that enforced coverage
cannot tell a unit test from a characterization golden, so the numbers describe how much code is
protected from change rather than how much is verified as correct.

**How to use this document.** It is a living artifact, meant for regression analysis (what was
supposed to catch this?), release-readiness review, onboarding, and planning CI work. The most useful
thing it can do is make the sentence "we assumed CI was covering it" less common. Every gap recorded
below is a place where that sentence would otherwise get said later, with worse consequences.

## Contents

Six sections carry the load, marked **P1** below, matching the priority the ROCm template assigns
them. If you are only reading part of this, read those. The short version above and this list are
additions to the template; everything else follows it.

**Orientation**

- [The short version](#the-short-version)
- Incident write-ups, headed *How this one was learned*:
  - [Why the net came before the tests](#how-this-one-was-learned-why-the-net-came-before-the-tests)
  - [One number and three months](#how-this-one-was-learned-one-number-and-three-months)
  - [A naming drift silently dropped a working kernel](#how-this-one-was-learned-a-naming-drift-silently-dropped-a-working-kernel)

**What we test, and how**

- [Component Overview](#component-overview) **P1**
- [Development Workflow](#development-workflow) **P1**
- [Testing Strategy and Layers](#testing-strategy-and-layers) **P1**
  - [Unit Testing Strategy](#unit-testing-strategy)
  - [Integration Testing Strategy](#integration-testing-strategy)
  - [Build-Time Validation of Library Logic](#build-time-validation-of-library-logic)
  - [Performance and Benchmarking Testing](#performance-and-benchmarking-testing) **P1**
- [Pre-submit / CI Gates](#pre-submit--ci-gates) **P1**
  - [Where these tests actually run](#where-these-tests-actually-run)
  - [Validation Gates and Ownership](#validation-gates-and-ownership)
  - [PR Test Classification](#pr-test-classification)
  - [Flaky Test Policy](#flaky-test-policy)
  - [Known Bugs and Expected Failures](#known-bugs-and-expected-failures)
  - [Coverage](#coverage)
  - [Nightly Validation](#nightly-validation)
  - [Supported Configurations](#supported-configurations)
- [Sanitizer Coverage](#sanitizer-coverage) **P1**
- [Static Analysis](#static-analysis)

**If you are writing a test**

- [For New Contributors](#for-new-contributors)
  - [Choosing the Right Test Type](#choosing-the-right-test-type)
- [Coverage Expectations by Change Type](#coverage-expectations-by-change-type)

**Why it looks like this**

- [Why We Test This Way](#why-we-test-this-way)
- [Key Quality Concerns](#key-quality-concerns)
- [Release Validation](#release-validation)
- [Dependencies and Validation Handoffs](#dependencies-and-validation-handoffs)

**What we owe**

- [Improvement Roadmap](#improvement-roadmap)
- [Known Risks and Gaps](#known-risks-and-gaps)
- [Owners and Review Cadence](#owners-and-review-cadence)

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

Four layers carry the load, and it helps to know what each one buys before reading any of them in
detail. **Unit tests** cover logic that needs no GPU, which in practice means TensileLite Python,
since the C++ side is thin for structural reasons named below. **Integration tests** are where nearly
all correctness confidence comes from: real GEMMs on real hardware, checked against a CPU reference.
**Build-time validation** checks the tuning data that drives kernel selection, and it runs inside the
build rather than as a test. **Performance testing** measures on every pull request and judges
nothing.

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
| Coverage | `tox -e coverage-unit` measures; `tox -e coverage-gate` enforces the floors against what it wrote |

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

#### How this one was learned: why the net came before the tests

> [!IMPORTANT]
>
> **The generator that emits every GEMM kernel takes a bug fix every four days on median, and those
> fixes keep landing on the same few concepts; that is what code with no testable seam does to the
> people maintaining it, and the goldens exist to create the seam, as scaffolding with a demolition
> date rather than a destination.**
>
> <details>
> <summary>The full account: the churn, the deadlock, and the plan that makes the net unnecessary</summary>
>
> [`KernelWriterAssembly.py`](tensilelite/Tensile/KernelWriterAssembly.py) is where a GEMM kernel
> actually becomes assembly, which makes it one of the highest-consequence files in the repository. It
> is 20,259 lines. Since the monorepo reorganization in April 2025 it has taken 303 commits from 78
> authors, adding 13,810 lines and removing 7,277. That window undercounts the file, which is years
> older than this repository. Roughly one commit in four is a fix, 74 of them, and the median gap
> between one fix and the next is four days.
>
> The fixes are not scattered. They cluster on a few ideas that keep coming back. Register lifetime is
> the worst by a distance, with 16 separate fixes to SGPR and VGPR allocation, release, alignment and
> overlap. Four of those land in a single nine-week run: not releasing certain SGPRs in one TDM kernel
> variant, then releasing them correctly in another, then releasing them before a StaggerU boundary,
> then a fourth on instantiation failures in a third variant. After that come tail-loop handling,
> StreamK and StaggerU work distribution, and sparse metadata at seven fixes each, then LDS layout,
> addressing and descriptors, and prefetch scheduling at six each. There are seven reverts. One had to
> be done by hand, deleting 472 lines, because the change had drifted too far to revert mechanically.
> Another was reverted the same day it landed and reapplied five days later.
>
> It is worth being careful about what that history means, because the obvious reading is the wrong
> one. Read the fix list and what you actually see is people repeatedly getting hard things right in a
> file where the language offers them no help: registers allocated by hand with no type to check the
> arithmetic, assembly built up as text, and correctness observable only on hardware.
> `KernelWriterAssembly.py` holds a single class with 287 methods and 290 distinct instance
> attributes. `KernelWriter.py` has one method 3,284 lines long. `mfmaIter` contains a single
> 34-branch `if`/`elif` chain. Landing a correct change in that requires holding more state in your
> head than anyone should be asked to hold, and that 78 people have moved it forward with no more
> breakage than this is the impressive part. The churn is a property of the code's shape. Change the
> shape and the churn goes with it.
>
> That is the case for refactoring, and it runs straight into a deadlock. The campaign's own
> assessment (AIHPBLAS-3865) states it plainly: the codegen path is simultaneously the least tested,
> at about 22.5 percent line and branch coverage under unit, and the most complex, with a file-average
> cyclomatic complexity of 16 to 27 and a worst function at 815 by that assessment's tooling. You
> cannot safely unit-test or refactor code in that shape without first recording what it does. The two
> halves hold each other in place: there is no seam to write a unit test against, and no safe way to
> create the seam without tests.
>
> The characterization suite is how that deadlock breaks. Photograph the behavior first, including the
> bugs, which are flagged in `DECISIONS.md` rather than quietly corrected. Reshape the code behind the
> photograph, so that any unintended change shows up as a reviewable diff. Then replace the
> photographs with real unit tests as each piece finally becomes testable. Every step of that order is
> forced by the situation rather than chosen.
>
> (One thing the history shows in passing: most of those 74 fixes carry no tracker reference, so the
> defect record for this file has largely lived in commit subject lines. That is actively being
> improved by the PR description and traceability hygiene now applied at review time and by the PR
> bot. It is a repository-wide practice rather than a testing question, so this document does not
> pursue it.)
>
> Two things have to stay true for any of this to be worth the trouble. The net has to stay honest,
> which is why blanket regeneration of the goldens is forbidden rather than merely discouraged. And
> something has to show that the net would actually notice a change, which today is mutation testing
> and nothing else. The plan then runs in phases: land the net, lock it with the coverage floor and
> snapshot governance, widen mutation beyond the pilot, make the codegen goldens portable across
> architectures, and finally refactor, each function graduating to real unit tests and losing its pins
> as it goes. Until that last phase lands, every coverage percentage in this document describes how
> much code is protected from change, not how much is known to be right.
>
> </details>

How that migration gets measured, and why no enforced number can see it, is covered under
[Coverage](#coverage).

The discipline that goes with goldens is documented in the suite's
[README.md](tensilelite/Tensile/Tests/unit/characterization/README.md) and is worth stating here
because it is easy to get wrong: **never run a blanket `pytest --snapshot-update`.** It rewrites
every golden at once and produces a green run that proves nothing. Update the smallest node id you
intend to change, read the resulting diff, and explain the behavior change in your PR description.

**The goldens are enforced, and they do gate a merge.** Which lanes assert them, and the one lane
that skips them, is covered under
[Where these tests actually run](#where-these-tests-actually-run).

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
written.

Today it is a report-only pilot on an eight-file slice, run through `tox -e mutation-unit` and
configured in `pyproject.toml`. It started at five files in
[PR #7989](https://github.com/ROCm/rocm-libraries/pull/7989) and grew by three in
[PR #9337](https://github.com/ROCm/rocm-libraries/pull/9337).
It is not a gate and does not run in CI. Accepted equivalent mutants
and every `# pragma: no mutate` are justified in `DECISIONS.md`. A series of PRs widening the
mutation-hardened surface starts at
[PR #10133](https://github.com/ROCm/rocm-libraries/pull/10133); those are still in draft.

#### Logic-corpus consistency regression tests

A third category sits outside the unit/characterization split: regression tests that scan the
*entire production logic YAML corpus* for cross-file naming and metadata invariants, rather than
exercising a fixture or pinning current behavior. Two files carry this today:
[`test_PlaceholderMerge.py`](tensilelite/Tensile/Tests/unit/test_PlaceholderMerge.py) (sibling
`DeviceNames` consistency, `_ID<chipid>` placeholder-suffix gating) and
[`test_GpuRevisionTarget.py`](tensilelite/Tensile/Tests/unit/test_GpuRevisionTarget.py) (the
gfx1250 v0/v1 overlay's logic-tree shape, 4 more tests behind the same gate). In spirit this is
closer to [`TensileLogic --check-all`](#build-time-validation-of-library-logic) than to a unit
test: both validate tuning data rather than code. The difference is placement.
`TensileLogic --check-all` is a mandatory build step that cannot be skipped. These are pytest tests
gated on whether the raw corpus
(`library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full`) happens to be present, which it is
not everywhere these tests run (see
[Known Bugs and Expected Failures](#known-bugs-and-expected-failures) and
[CI visibility and gating](#ci-visibility-and-gating)).

#### How this one was learned: a naming drift silently dropped a working kernel

> [!IMPORTANT]
>
> **A naming rule that quietly fell out of step with a runtime hardware check made the library
> throw away a working kernel with no error at all, and gfx942 users found out only when real GEMM
> calls stopped finding any kernel to run.**
>
> <details>
> <summary>The full account: how a filename drift turned into a missing kernel</summary>
>
> In February 2026 ([PR #6946](https://github.com/ROCm/rocm-libraries/pull/6946), ROCM-23637), GEMM
> calls on gfx942 started failing outright with "no solution found," for problem sizes that had
> worked before. Nothing crashed and no test caught it in advance.
>
> hipBLASLt decides which compiled kernel to run in two places that are supposed to always agree
> with each other: a runtime check of which GPU chip is actually in front of it, and a naming
> convention baked into the kernel files when the library is built. A recent change had let those two
> drift apart for one family of kernels. The runtime check still considered two kernel files
> identical, but the build now gave them different names. The step that assembles the final library
> trusted the names over the runtime check, treated the two as unrelated, and quietly kept only one
> of them. No error, no log line, just a kernel that had existed a moment before and no longer did.
>
> The fix closed both ends: it put the naming rule back in step with the runtime check, and it
> straightened out the affected kernels' metadata so that files describing the same kernel agreed
> with each other again. It also added `test_PlaceholderMerge.py`, so that a naming/runtime
> disagreement like this one gets caught immediately instead of surfacing as a mysterious missing
> kernel later.
>
> </details>

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

**Suite size.** The characterization half is about 2,891 tests and the pure-unit half about 3,126,
of which roughly 230 are GPU-guarded and skip on a CPU-only runner. Call it six thousand tests, and
note that all of them run in four separate CI lanes (see
[Where these tests actually run](#where-these-tests-actually-run)).

**Coverage expectation.** The enforced whole-project floor for TensileLite Python is `fail_under = 75`
in [`tensilelite/pyproject.toml`](tensilelite/pyproject.toml), which is deliberately the only place
that number lives. The comment beside it records the intent: 80% is the target, and the floor is set
below the measured value on purpose so that ordinary run-to-run noise cannot trip an exact cutoff.
Recent runs measure 78.55% on the GitHub Actions lane and 78.68% in Math CI, which is close enough
agreement to trust. Per-file floors ratchet separately.

Read that number with the caveat above firmly in mind: it is union coverage across the unit and
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

**What it does.** `TensileLogic --check-all` validates the library logic YAML before any of it is
compiled. It checks chip IDs, matrix instructions, work-group shapes, the XCC work-group mapping, and
custom kernel declarations, one file at a time. It reads YAML only, so it needs no GPU and no compiled
kernels, and it is fast. A failure stops the build. It is the only mechanism in the component that
validates tuning data rather than code, it does not look or report like a test, and it exists because
of one specific incident.

#### How this one was learned: one number and three months

> [!IMPORTANT]
>
> **One retuned number in a YAML file silently disqualified every kernel candidate on a 38-CU
> partition, the fallback path hid that as a 3x slowdown rather than a failure, and it cost the better
> part of three months; the validator that would have caught it in seconds was already sitting in this
> repository, wired to nothing.**
>
> <details>
> <summary>The full account: how it was found, why nobody caught it, and what the fix actually took</summary>
>
> In February 2026, someone investigating a slow inference workload on an MI300X found that ROCm 7.2
> took three times as long as ROCm 7.0 to do the same work. Nothing crashed. No answer was wrong. It
> was just slow. They minimized it to three back-to-back matmul calls, and the numbers were stark: 2.6
> seconds became 8.5, and the count of `hipGetDeviceProperties` calls went from 2,594 to 66,189.
> Twenty-five times the driver traffic for three matrix multiplications. Swapping the caller's BLAS
> backend made it vanish, which is what turned it into a hipBLASLt ticket, ROCM-2963.
>
> The cause was one number in a YAML file. The GPU was in CPX mode, which partitions it into eight
> units of 38 compute units each. Kernel selection has a predicate, `WorkgroupMappingXCCCheck`, that
> requires the CU count to divide evenly by a solution's `WorkGroupMappingXCC`. Somewhere between 7.0
> and 7.2, that value was retuned from 1 to 4 in the 38-CU library. 38 is not divisible by 4. Every
> solution failed the predicate, the heuristic returned zero candidates, and the library fell back to
> `getAllSolutions`, an exhaustive scan of roughly 2,680 candidates, on every single call.
>
> Notice that the fallback worked. It did exactly what it was designed to do, and that is precisely why
> nobody caught this: the safety net converted a total kernel-selection failure into a performance
> problem, and performance has no gate in this component (see
> [Performance and Benchmarking Testing](#performance-and-benchmarking-testing)). So it shipped.
>
> Then there is the scale. The fix, [PR #5009](https://github.com/ROCm/rocm-libraries/pull/5009),
> changed 15,841 lines across 21 YAML files, and every one of those lines was the same edit:
> `WorkGroupMappingXCC: 4` back to `1`. Nearly sixteen thousand solutions, every one syntactically
> valid, every one reviewed and merged, every one silently unusable on the hardware that library exists
> to serve. Nothing in the component could have caught it, because nothing in the component was looking
> at the data at all.
>
> The fix then had a rough month. It merged, was cherry-picked into the 7.2 release branch, was
> reverted, was proposed for revert again, and was reverted a second time when the 7.2.1 release team
> decided not to take it. That part never shows up in a root-cause summary. One retuned parameter
> produced the better part of three months of tickets, cherry-picks, reverts and meetings between the
> first symptom and a guardrail that would have caught it in seconds.
>
> Here is the part that should change how you read the rest of this document: most of what the fix
> needed already existed. `TensileLogic` had been in the tree for over a year, with validators for
> work-group shapes and matrix instructions already written. Somebody had been bitten by bad logic data
> before and had built a tool to catch it. But nothing ran that tool. It was wired into neither the
> build nor CI, and it had drifted out of step with the data it was meant to check.
>
> So [PR #5039](https://github.com/ROCm/rocm-libraries/pull/5039) was less about writing a checker than
> about making the existing one bite: clean it up, add a `WorkGroupMappingXCC`-versus-CU validator for
> the rule that had actually been violated, and wire the whole thing into the build ahead of codegen so
> invalid logic cannot reach a `.dat` file. It also added a validation API in TensileLite encoding the
> rules a solution must satisfy to be selectable, with unit tests that inject a CU count of 38 directly
> and assert that `WorkGroupMappingXCC: 4` fails and `1` passes. That is the original three-month bug,
> reproduced in milliseconds on a CPU, with no CPX-mode hardware anywhere in the loop.
>
> The uncomfortable implication is that ROCM-2963 was probably not the first time this class of defect
> cost somebody weeks. It is the first time it was well enough documented to point at. A validator
> nobody runs is indistinguishable from no validator at all, and this document names three more of
> them: `tox -e lint`, configured and invoked by no CI job; a CodeQL workflow that never sees C++ and
> never runs on a pull request; and TSAN build options that exist while no lane uses them. Each one is
> a tool somebody wrote because they had been burned, sitting where it cannot burn anything back.
>
> </details>

#### Why it runs in the build

Relative to the build, the logic YAML is compiler input rather than build output:
`TensileCreateLibrary` consumes it and emits kernels from it. Validating it is front-end analysis
rather than testing, and running codegen over input already known to be invalid produces output nobody
should trust. So the check is wired in as a CMake custom command in
[`cmake/HipBLASLtCodegen.cmake`](cmake/HipBLASLtCodegen.cmake) that runs ahead of
`TensileCreateLibrary` and writes a stamp file. A failure stops the build.

That placement buys good reach. It runs on every build that generates kernels, including every
developer's local one, so a bad entry surfaces in the edit-build loop instead of a CI round trip
later. It cannot be skipped by a path filter and it cannot be dropped by a gating rule, neither of
which is true of the gating `preliminary` job.

That reach used to extend past the build's own target architectures. From when this gate was wired
into the build ([PR #5039](https://github.com/ROCm/rocm-libraries/pull/5039), 2026-05-05) until
2026-08-13, the build-wired invocation validated every logic file in the corpus regardless of which
`GPU_TARGETS` were being compiled, so a developer building only gfx1151 would still notice a broken
gfx942 entry.
[PR #9218](https://github.com/ROCm/rocm-libraries/pull/9218) (merged 2026-08-13) removed that:
whole-corpus validation was dominating incremental build time on single-arch builds (roughly ten
minutes locally for a gfx1151 build, validating the full multi-arch logic set when only a few dozen
files were relevant), so the CMake step now passes `--architecture "${GPU_TARGETS}"` to
`TensileLogic`, and only logic files matching the build's own target architectures are checked.
`TensileLogic --check-all` still defaults to validating the whole
corpus when a developer runs it by hand with no `--architecture` argument; it is only the build-wired
invocation that is now scoped down. The trade was deliberate and reasonable for build time, but it is
worth naming plainly: a single-architecture CI build no longer catches a broken entry in an
architecture it does not target, which narrows this gate's reach to exactly the union of architectures
across whatever set of builds happen to run on a given pull request.

What the placement costs is visibility and strictness. There is no check name in the pull request, no
test report, and no way for CI to run it in isolation. It also cannot be tightened in place, because
`--strict-known-bugs` would fail local developer builds over a stale entry someone else owns, which is
not a reasonable thing to do to a person trying to build. Both costs point at the same answer, and
that answer is a second lane rather than a different home for this one.

#### What it found once it existed

The first thing a new gate does is argue with itself, and this one was no exception. Pointed at the
full library it reported `Total 552358 solutions / Keep 552345 / Reject 13`. Thirteen gfx950 solutions
failing matrix instruction validation, quarantined rather than fixed and still open under ROCM-7144.
Shortly after, it failed a TheRock CI build on a gfx942 fp8 configuration, and that one turned out to
be the validator's fault rather than the data's: its MFMA tables were missing the fnuz key variants
(ROCM-24036, fixed in the same PR that added the gate).

Since then it has been quiet, which is the outcome you want and the hardest one to take credit for. A
retuning that breaks a CU-variant library the way ROCM-2963 did now fails a build in seconds instead
of reaching a customer's model three months earlier in the story. There is no way to know how many
times that has already happened, because a build that fails on line one of a bad YAML file does not
generate a ticket, a meeting, or a revert. That absence is the whole return on the investment.

Its known-bug list,
[`tensilelite/Tensile/TensileLogic/known_bugs.yaml`](tensilelite/Tensile/TensileLogic/known_bugs.yaml),
is the best-structured quarantine in the component (see
[Known Bugs and Expected Failures](#known-bugs-and-expected-failures)). Entries are keyed on the
logic file path plus the solution's `SolutionNameMin`, a content-derived name adopted in
[PR #9355](https://github.com/ROCm/rocm-libraries/pull/9355) so that keys survive library re-tuning
instead of drifting with a positional index, and each entry carries a `ticket:` field. The checker
re-validates every entry and reports the ones that no longer reproduce, so a fixed bug is detected
rather than skipped forever. All 14 current entries document the same gfx950 validation drift.

`--strict-known-bugs` turns that detection into a failure, but it defaults off and nothing passes it
today, so a stale entry only warns. Enforcing it is tracked in AIHPBLAS-4196, which proposes the
right fix: a dedicated GitHub Actions job running
[`scripts/run_tensile_logic_check.py`](scripts/run_tensile_logic_check.py) with the flag, rather than
tightening the in-build command. That ticket also names a remaining hole, which is that an orphaned
entry whose `solution_name` resolves to nothing currently matches nothing and is silently ignored, so
strict mode is not yet a complete dead-entry detector. Extending the gate to cover derived-parameter
assignment is tracked in AIHPBLAS-3575.

### Performance and Benchmarking Testing

Performance measurement for hipBLASLt runs in a nightly lane outside this repository. Pull requests
are not benchmarked. This section describes what that lane covers, how a number is produced, and how
large the measured sample is. What we would like to add is in the
[Improvement Roadmap](#improvement-roadmap).

| Item | Detail |
| --- | --- |
| Stack layer | Core SDK (math library) |
| Metrics measured | GFLOP/s, memory bandwidth (GB/s), latency (µs). Optionally clock frequencies, achieved efficiency, and memory read/write bytes |
| How benchmarks are run | `hipblaslt-bench` for a single problem or a YAML batch; `hipblaslt-perf` to run named suites and aggregate CSV over repeated samples; `hipblaslt-cotenant` to measure under CU contention |
| Where measurement runs | A nightly lane defined in an AMD-internal repository named `rocPTS`, owned by the performance tracking team. No lane defined in this repository benchmarks anything, and `hipblaslt-perf` has no caller in this tree |
| Automated runs | The rocPTS nightly. No other automated benchmarking runs at any cadence |
| Sizes measured | About 76,800 problems a night on gfx950. See [How many sizes we measure](#how-many-sizes-we-measure) |
| Reference for comparison | The nightly compares each result against a rolling baseline of recent runs on the same device |
| Where results are stored | Ingested into rocPTS, AMD's performance tracking service, and viewable from a Kibana dashboard and the PTS web app. Local runs write CSV to the developer's workspace |
| Regression threshold | Not defined in this repository. The nightly applies one and reports to its owners. The tuning flow's uplift threshold is a candidate filter rather than a regression gate |
| Gating approach | No merge gate on performance |

#### Measurement on pull requests

Pull requests are not benchmarked. No performance number is produced for a change, so there is
nothing to compare against a baseline and nothing to threshold at review time.

Two files in the tree read as though a per-pull-request lane consumed them.
`clients/scripts/performance/problems/matmul_probset1_bench.yaml` holds 35 shapes, NN only, with one
mixed FP8 type, and the `ci_perf_job` suite is defined here as well. Neither has a caller in this
repository today.

#### The nightly measurement lane

The lane is defined outside this repository, so it is described here for contributors who would
otherwise have no way to find it. Benchmarking for hipBLASLt runs nightly from `rocPTS`, an
AMD-internal repository owned by the performance tracking team. Nothing in this repository triggers
it or reports its results.

What it covers, as of 11 August 2026:

- **Architectures.** hipBLASLt is scheduled on two gfx950 devices, MI350 and MI350P. The MI350P leg
  has been cancelled before starting on recent nights, so one device is measured in practice. Other
  architectures the library ships tuning data for are not benchmarked on a cadence.
- **Sizes.** About 76,800 problems a night, from four suites that are version controlled in that
  repository and reviewed like code. See [How many sizes we measure](#how-many-sizes-we-measure) for
  the breakdown.
- **Comparison.** Each result is compared against a rolling baseline of recent runs on the same
  device, with a threshold and an alert to the lane's owners. Results are not surfaced on a pull
  request.

Two properties follow from how the lane is scheduled. It measures a nightly build of the whole
monorepo rather than an individual commit, so a change is first measured the night after it merges
and the comparison spans every change that landed in that window; a movement is attributed to a day
rather than to a pull request. And the alert goes to the lane's owners, so the reader of a result is
usually not its author.

#### How a number is produced

Useful background for judging whether a performance delta is real. No lane defined in this
repository produces a performance number, so a contributor watching the GitHub checks list will not
see one. What follows is how the tools in this tree behave when a developer or a lane runs them.

`hipblaslt-perf` invokes `hipblaslt-bench` once per problem per outer sample, five outer samples by
default, and reports the mean and the median across those samples along with every raw value. Within
one invocation, `hipblaslt-bench` runs `cold_iters` untimed launches, then times a batch of `iters`
launches and divides, so a reported figure is a per-iteration average over a batch of back-to-back
enqueues rather than a single timing. `rotating: 512` allocates as many copies of the problem's
buffers as fit in 512 MiB and cycles through them by iteration, so those iterations cannot all read a
warm cache. The nightly lane does not use `hipblaslt-perf` at all: it calls `hipblaslt-bench` once per
problem with an iteration count carried in the dataset, and gets its stability from comparing against
past nights rather than from repeating within a night. A single night's number for a single shape
therefore carries less information than its trend.

Two caveats apply to any of these numbers. The harness has an adaptive mode that samples until the
relative standard error of the mean falls below one percent, or failing that until a robust dispersion
measure plateaus within five percent, and no lane uses it: `adaptive` defaults to false and nothing
turns it on. Iteration counts are fixed ahead of time rather than chosen by a stability criterion,
and the output does not report whether a given number converged.

The second concerns the machine. This repository contains no clock pinning, no performance-level
setting, no CPU affinity or NUMA pinning, and nothing that keeps another workload off the GPU while
a measurement runs. The only isolation is that measurement is scheduled onto dedicated benchmark
nodes. Performance level and clocks are recorded next to the results, and the harness can report
achieved clocks when an environment variable is set, but no lane sets it, so clock drift enters the
number as noise rather than being visible as a fact about the run. Whether those nodes are quiet in
practice is a property of the CI system rather than of this repository, and a reader of this
repository cannot verify it. Small differences should therefore be interpreted with care.

#### How many sizes we measure

The problem space is effectively unbounded: on one gfx950 device, counting only shapes whose
dimensions are multiples of 16 and whose operands fit in memory, the number of distinct FP8 batched
GEMM problems is on the order of trillions. No suite samples that by enumeration, so what matters is
how the sample was chosen and which axes it holds fixed.

- **Pull requests.** Not benchmarked; see
  [Measurement on pull requests](#measurement-on-pull-requests).
- **Nightly, gfx950** (the rocPTS lane): **76,768 problems** from four version-controlled suites. One
  is a model-shape set of 3,085 problems, mostly bf16 with the rest fp32, spanning all four transpose
  combinations and holding the only batched problems in the set. The other three are large TN sweeps
  of a single data type each: 24,629 in bf16, 24,629 in FP8, and 24,425 in TF32. Counted on 7 August
  2026, with no dataset change since. The set grew roughly fortyfold during July 2026, so the number
  is a snapshot rather than a fixed fact.

**Two axes the sample holds fixed.** Of the 76,768 nightly problems, 200 have a batch count above
one, all of them in the model-shape set and all bf16 or fp32, because the three large sweeps are
non-batched; no batched FP8 problems are measured at any cadence. Leading dimensions are the second
axis: every problem uses the natural minimum stride for its layout, so measurement does not cover
padded strides. Kernel performance is known to depend on leading dimensions, and the shipped tuning
data keys on them for part of its coverage (see
[Tuning coverage](#tuning-coverage-is-a-different-number-and-it-is-measured-by-hand)). Coverage is
therefore broad along M, N and K for four data types in TN, and fixed along batch and stride.

**Gating summary:**

| Gating level | Status | Notes |
| --- | --- | --- |
| PR-level automated measurement | **No** | Pull requests are not benchmarked |
| PR-level automated gate | **No** | No per-PR number exists to gate on |
| Nightly measurement outside this repository | **Yes** | About 76,800 problems on gfx950 in the rocPTS lane, most nights |
| Automated regression alerting | **Yes, outside this repository** | The nightly compares against a rolling baseline and alerts its owners. Results are not surfaced on a pull request, are not attributed to a single change, and do not block a merge |
| Release qualification | **No documented gate** | Performance is discussed at release time but there is no in-repo criterion |

#### Tuning coverage is a different number, and it is measured by hand

Tuning is where most sizes actually get benchmarked, and it is worth separating from CI because the
two get quoted as one thing. The shipped library logic for gfx950 carries 21,706 exact
problem-to-solution entries covering 8,903 distinct (M, N, batch, K) points, of which 4,116 entries
and 2,231 distinct sizes are FP8 TN. Each of those points was benchmarked at least once, by the
tuning flow, to decide which kernel wins. Notably, 2,982 entries carry an eight-element key that
includes leading dimensions, so the tuning data already treats stride as selection-relevant even
though no benchmark lane varies it.

That work is manual and campaign-driven rather than continuous. It arrives as human-authored pull
requests, one per data type, layout and device variant, usually carrying a ticket: the gfx950 logic
alone has taken 217 commits since April 2025, with subjects of the form *Tuning Equality BBS F8BS for
gfx950_id75a3* and *7 GEMM sizes added for gfx950/75a3 bbs tn*. Nothing schedules it, nothing in this
repository triggers it, and no job re-measures those points afterward. In summary, roughly nine
thousand sizes on this architecture were each measured once, by hand, to choose a kernel. Whether the
nightly's sweeps re-cover any of them is not tracked anywhere.

Related tooling is sometimes mistaken for regression testing. `utilities/geko/` is a GEMM kernel
optimizer that searches for better kernels and benchmarks candidates during tuning, and
`utilities/QuickTune/` performs offline per-workload tuning from captured logs. Both produce
performance numbers as part of tuning rather than as a regression signal, and neither has a CI caller.
Similarly, TensileLite's benchmark and library-logic phases generate performance data used to *select*
kernels.

**Known gaps:** pull requests are not benchmarked, so a movement cannot be attributed to the change
that produced it; the nightly narrows it to a day of merges. No threshold or gate is applied in this
repository, and there is no gate on library size, kernel count, or build time. Measurement runs
without clock pinning or any machine-quieting step, and the harness's stability criteria are not
enabled, so run-to-run noise is neither bounded nor reported. Leading dimensions are not varied and
batched FP8 is not measured. Benchmark coverage is gfx950, narrower than the set of architectures the
library ships tuned kernels for. Today a regression surfaces through the nightly alert, through
someone reading the dashboard, or through a downstream consumer. What we would like to add is listed
in the [Improvement Roadmap](#improvement-roadmap).

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
- **`static-analysis`** is not what its name suggests. It scans the working tree and the git log for
  a list of sensitive words maintained in the CI system rather than in this repository, and fails on
  any match. It is a disclosure gate protecting a public repository, not a code-quality analyzer. See
  [Static Analysis](#static-analysis) for what that means in practice.
- **`preliminary`** is the functional gate for TensileLite, and it runs two stages on gfx12, gfx90a,
  gfx942, and gfx950. First `tox -e py3 -- Tensile/Tests "-m common"`, the GEMM selection that
  genuinely needs hardware. Then, *only if that stage passed*, `tox -e unit -- Tensile/Tests/unit`,
  which carries no marker filter and therefore runs the entire unit tree, characterization included,
  on all four architectures.

  Three conditions narrow when any of this happens, and all three are easy to miss:

  1. The job diffs the change against `develop` and skips entirely when nothing under
     `tensilelite/`, `shared/stinkytofu/`, or `shared/origami/` has changed. A pull request touching
     none of those passes it without running anything.
  2. The second stage is conditional on the *target branch*. It runs only when the PR targets
     `develop` or one of the two `hipblaslt_common_cms_*` branches. Which tests gate your change
     depends on where you aimed it.
  3. The gate itself can be dropped. Math CI's `statusGate` carries a rule that removes `preliminary`
     from hipBLASLt's gating list whenever the same pull request also touches rocroller, so a
     rocroller-touching change loses this gate silently.

Other Math CI jobs post checks without gating. The one worth knowing is
`tensilelite-unit-codecov`, which runs the TensileLite Python and C++ coverage environments on gfx950
and uploads to codecov under the `TensileLite-Unit` and `TensileLite-CPP` flags. It is two jobs in
one, a Python coverage run and a C++ coverage run sharing a single script, which is why it needs
gfx950 and why it takes hours rather than minutes. It reports a check, but that check is not
required, and a failure anywhere in it skips the codecov uploads, so a codecov number can be older
than the commit you are looking at.

### Where these tests actually run

The TensileLite Python tree under `Tensile/Tests/unit` (characterization *and* unit, about six
thousand tests) executes in four separate CI lanes. They are easy to confuse with each other, and
with three adjacent lanes that sound like them but run none of these tests, so the whole set is worth
laying out once.

| Lane | Where defined | Hardware | Gates? |
| --- | --- | --- | --- |
| `Component CI: TensileLite coverage` | [`component-ci-tensilelite-coverage.yml`](../../.github/workflows/component-ci-tensilelite-coverage.yml) | CPU only | No. Rolls up to `Component CI Summary`, which is not required |
| `preliminary` | Math CI (internal) | gfx12, gfx90a, gfx942, gfx950 | **Yes**, via the required `Math CI Summary` |
| TheRock `Test tensilelite` | [`test_tensilelite.py`](https://github.com/ROCm/TheRock/blob/main/build_tools/github_actions/test_executable_scripts/test_tensilelite.py) in TheRock | GPU runner, Linux | **Yes**, via the required `TheRock CI Summary` |
| `tensilelite-unit-codecov` | Math CI (internal) | gfx950 | No |

Three observations follow from that table, and they are the ones that most often get stated
backwards in review:

**The tests gate; the coverage numbers do not.** Both required checks (`Math CI Summary` through
`preliminary`, and `TheRock CI Summary` through the TensileLite job) execute the unit and
characterization suites, so a broken test blocks a merge. Neither of them looks at coverage. The
floor and the per-file ratchet are enforced only in lanes that cannot block anything. "None of it
gates" is wrong, and so is treating a coverage regression as a merge blocker.

**Three of the four lanes hold a GPU, and only one of them needs it for these tests.** TheRock's lane
is validating a real install, so its GPU is the point. `preliminary` needs hardware for its
`-m common` GEMM stage and the unit tests inherit the node because they are appended to the same
script. `tensilelite-unit-codecov` needs gfx950 for its C++ half only; the Python coverage run is
along for the ride, which is most of why that job takes about two and a half hours where the
CPU-only GitHub Actions lane takes seven minutes.

**Every one of these lanes is fail-fast, and that erases signal.** In each case characterization-ish
work runs first and everything after it is conditional on it passing: the two coverage lanes through
the ordering of tox `commands`, `preliminary` through an explicit exit-code guard, TheRock's through
plain sequential ordering. One early failure means you get no information at all from the stages
behind it, which is why a red run so often says less than it appears to.

**The goldens are asserted in three of the four lanes,** because every tox environment involved
inherits syrupy from the base `[testenv]` dependency list. `preliminary` is the one that matters,
because it gates: its second stage runs the tree with no marker filter, so a stale golden fails a
required check. The two coverage lanes assert them as well and cannot block a merge, though the
CPU-only GitHub Actions lane is fast enough that it is usually where a stale golden surfaces first.

TheRock's installed-artifact lane is the exception. It does not install syrupy, so the suite's
`conftest.py` detects the missing plugin and skips the snapshot-using tests cleanly rather than
erroring the whole run. The tests are still collected, which is why that lane's log reads as though
characterization ran. Nothing is lost, because those goldens were asserted in the source lanes before
the artifact was built, but a reader scanning that run will see skips and should know they are
deliberate.

Three adjacent lanes run none of these tests, despite their names:
`Component CI: rocISA` only tests a `pip install` of the rocisa package; Math CI's `codecov` job is
C++ hipBLASLt coverage and is one character away from `tensilelite-unit-codecov`; and `precheckin` is
the C++ build and test.

### Validation Gates and Ownership

| Validation area | Required before merge | Owner | Responsibility |
| --- | --- | --- | --- |
| Build (Linux and Windows) | Yes | CI / DevOps + TheRock | Maintain jobs, runners, and pipeline health |
| Library logic validation (`TensileLogic --check-all`) | Yes, implicitly | Component team | Runs inside the build ahead of codegen, so it blocks any kernel-generating build. No check name, no test report |
| Unit tests (TensileLite Python) | Yes | Component team | Create, maintain, review |
| Integration / smoke tests (client GTest) | Yes | Component team | Validate behavior across key scenarios |
| Characterization goldens | Yes, when `tensilelite/` is touched and the PR targets `develop` | Component team | Asserted by the gating `preliminary` job. Review every golden diff; never bulk-regenerate |
| HOST_ASAN build and quick test | Yes, on gfx90a | Component team | Keep the sanitizer lane green |
| Code coverage floor and ratchet (TensileLite) | **No** | Component team / CI | Enforced, but only inside lanes that roll up to non-required checks. Floors move up only, on the honor system |
| Formatting and lint (`pre-commit`) | Yes | CI / DevOps | Maintain hooks |
| Sensitive-word scan (the Math CI job named `static-analysis`) | Yes | CI / DevOps | Gating, but it is a disclosure gate rather than code analysis |
| Code-quality static analysis (C++) | No | Unowned | **Nothing runs.** No clang-tidy or cppcheck configuration exists for hipBLASLt, and CodeQL does not cover C++ |
| Code-quality static analysis (Python) | No | Component team | `tox -e lint` exists but no CI job invokes it, and it is narrowed to pyflakes checks |
| Client test suite on gfx90a / gfx942 / gfx950 / gfx12 | Yes | Component team | Gating Math CI job (`precheckin`) |
| Performance | No | Component team | Pull requests are not benchmarked. Benchmarking runs nightly, outside this repository (see above) |
| Shared validation infrastructure | N/A | TheRock team | Provide shared build and test infrastructure |
| System validation | N/A | QA | Execute system-level and release validation |
| Release qualification | N/A | Component team + QA + TPM | Confirm readiness, review known risks |

A caveat on this table: the Math CI rows are taken from that system's gating configuration, which is
authoritative but lives outside this repository. Which GitHub Actions checks are configured as
*required* in branch protection is not documented anywhere a contributor can see, and the answer is
not consistently understood even among the people working on the tests. Publishing the actual
required-check list, in this repository, is tracked as a gap.

### PR Test Classification

**Trusted gate.** A failure here is a real problem with the change.

- Build on both platforms, which includes the `TensileLogic --check-all` library logic validation
- Client GTest quick tier on gfx94X-dcgpu (Linux) and gfx110X (Windows)
- The Math CI client suite on gfx90a, gfx942, gfx950 and gfx12
- The Math CI sensitive-word scan, the job named `static-analysis`
- TensileLite Python unit and characterization suites: on GPU across four architectures via
  `preliminary`, and against installed artifacts via TheRock
- HOST_ASAN build and quick test on gfx90a
- `pre-commit`

**Informational.** Worth reading, cannot block a merge.

- HOST_ASAN on gfx942, opt-in via the `ci:asan` label and explicitly non-blocking
- The TensileLite coverage floor and per-file ratchet, which fail their own lane but not a required
  check
- The characterization-versus-unit coverage summary card
- The `tensilelite-unit-codecov` check and the codecov reports

**Unstable / flaky.** Not tracked as a category; hipBLASLt has no `UNSTABLE` tag. That is a real gap
rather than a claim that nothing is flaky. What exists instead is a quarantine list, described next.

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

hipBLASLt suppresses or records known-bad behavior in eight different places. They accumulated
independently, they use different formats, and they are not governed as one thing. Anyone reasoning
about "what do we currently know is broken" has to check all eight.

| Mechanism | What it suppresses | Ticket linkage | Detects its own fix? |
| --- | --- | --- | --- |
| [`clients/tests/data/known_bugs.yaml`](clients/tests/data/known_bugs.yaml) | Client GTest cases matched by parameters, optionally per architecture. Excluded from every tier | Comment convention | **No.** The case never runs, so nothing can observe a fix |
| `GTEST_SKIP()` in client sources | Individual cases at runtime | None | Not applicable, and mostly not bugs: these are environment guards (no GPU present, no Stream-K kernel selected for the problem) |
| [`TensileLogic/known_bugs.yaml`](tensilelite/Tensile/TensileLogic/known_bugs.yaml) | Library-logic validation failures, keyed on logic file path plus `SolutionNameMin` | Structured `ticket:` field | **Partly.** Re-validates each entry and reports stale ones, but only warns |
| Filename-driven marks in `Tensile/Tests/common/config_helpers.py` | Any config YAML whose path contains `xfail`, `wip` or `disabled` | None; the reason lives in a filename | **No**, and non-strict, so an expected failure that starts passing is silent |
| `skip-<arch>` marks in config YAML `TestParameters` | A config on named architectures | Free-text comment | Not applicable |
| Explicit `pytest.mark.xfail` markers | Specific assertions in a Python test | Ticket in the `reason` string | **Yes**, when written `strict=True` |
| Characterization goldens that pin known-wrong behavior | Nothing. The wrong behavior is recorded rather than hidden | ADR under `adr/` with a defect link, required by the reviewer checklist | Not applicable: a fix shows up as a golden diff needing review |
| `_needs_logic_dir` environment-conditional `pytest.mark.xfail` ([`test_PlaceholderMerge.py`](tensilelite/Tensile/Tests/unit/test_PlaceholderMerge.py), duplicated in [`test_GpuRevisionTarget.py`](tensilelite/Tensile/Tests/unit/test_GpuRevisionTarget.py)) | The logic-corpus consistency checks described under [Logic-corpus consistency regression tests](#logic-corpus-consistency-regression-tests), whenever `library/.../Logic/asm_full` is not on disk | Issue URL in the `reason` string; no `strict`, no time-box | **No.** The condition tracks an environment, not the bug it guards; where that environment is permanent (see below) the check can never run for real regardless of what the data says |

This last mechanism is a different shape from the other seven: it is not quarantining a *known* bug
at all, but gating on a precondition, and it lands in the same **Blind** tier as the client
quarantine list for a more permanent reason. In TheRock CI's installed-artifact layout, the corpus
this precondition checks for never exists by design (see
[CI visibility and gating](#ci-visibility-and-gating)), so the tests behind it (2 in
`test_PlaceholderMerge.py`, 4 in `test_GpuRevisionTarget.py`) cannot execute for real in that lane,
ever, independent of whether the underlying data is correct. [PR #7716](https://github.com/ROCm/rocm-libraries/pull/7716)
narrowed the marker from a module-wide xfail (which was false-XPASSing 3 unrelated tests) to just
the 2 tests that need the corpus; it fixed the XPASS problem it was solving but left this shape
intact.

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

**Known gaps.** Only one mechanism records its ticket somewhere a tool could read, and none records a
review date. Nothing reports what is currently suppressed across all seven places, or for how long.
The filename-driven marks in `config_helpers.py` are the weakest link by construction, since a path
substring cannot carry a ticket or a reason at all, and the resulting mark is non-strict so a fix is
invisible; they have no users in the tree
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

#### The characterization-versus-unit split

The enforced floors cannot see that migration at all. Coverage is measured on the union of the two
suites, so converting a line from characterization protection to a genuine unit test leaves every
enforced number identical. A file can sit at 80% while almost all of that 80% is scaffolding, and no
gate will say a word.

What shows the migration is the split summary card, rendered into the GitHub Actions run summary
by the coverage lane. It splits every measurable statement into four buckets that sum to 100%:
reached by both suites, by characterization only, by unit only, and by no test at all. The
characterization-only count is the migration debt, and the goal is for it to fall toward zero.

The card also ranks the largest files by statement count with each file's unit-suite and
characterization-suite percentages side by side, which is how the next refactor target gets picked: a
high characterization percentage next to a low unit percentage is a file still leaning on scaffolding.
The two suites are kept disjoint in the coverage lane specifically so this attribution means
something.

Worth being blunt about the consequence. The one number that measures real progress is the one number
nothing enforces. Reading a file's coverage without reading its split is how a team convinces itself
it is further along than it is.

#### What is measured today

| Scope | Tool | Measured in | Enforced |
| --- | --- | --- | --- |
| TensileLite Python, unit and characterization combined | `coverage.py` via `tox -e coverage-unit` | GitHub Actions, on any change under `tensilelite/**`; Math CI measures it again for codecov | Floor plus per-file ratchet, 1 pp tolerance. Enforced in a lane that is not a required check |
| TensileLite Python, unit-only share | Same lane, reported in the split summary card | GitHub Actions | **No.** Informational only, and it is the number that tracks real progress |
| TensileLite Python, mutation score | `tox -e mutation-unit` | Nowhere; run by hand | No. Report-only pilot on eight files |
| TensileLite C++ host library | `tox -e coverage-cpp` | Math CI | Reported to codecov, not enforced |
| hipBLASLt C++ library | Optional `HIPBLASLT_ENABLE_COVERAGE=ON` build | Not run in CI | No |

The GitHub Actions coverage lane is CPU-only and takes roughly seven minutes. It runs the
characterization suite and the pure unit suite once each under coverage, keeping the two selections
disjoint so each line can be attributed to one or both, unions the results, and renders the
non-gating split summary card. That lane is deliberately scoped and is expected to retire once the
characterization-to-unit conversion finishes, which makes the card's characterization-only count a
rough progress bar for the lane's own retirement.

#### Measuring and enforcing are separate tox environments

This is worth knowing before you try to reproduce a coverage failure. `coverage-unit` runs the suites
and writes the reports, reporting with `--fail-under=0` so it never gates. `coverage-gate` reads those
artifacts and applies both floors. The GitHub Actions lane runs them as two named steps and owns the
floors; Math CI's `tensilelite-unit-codecov` runs only the first, so it cannot be failed by a gate it
does not own.

The practical benefit is local reproduction. `coverage-gate` sets `skip_install = true` and depends
only on `coverage[toml]`, so it builds no rocisa, needs no ROCm, and runs no tests. Run
`coverage-unit` once, then re-run `coverage-gate` as often as you like against the `coverage.json`
you already have. Chasing a floor failure costs seconds rather than another pass over six thousand
tests.

#### Targets

Two different mechanisms carry a number, and they are easy to confuse:

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

#### Scope and exclusions

Coverage measurement is Python-focused, and measured on Linux only; Linux
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
- **TensileLite GPU tests.** The `Component CI: TensileLite coverage` lane is CPU-only, so its roughly
  230 GPU-guarded tests skip there by design. They do run per PR, on Math CI's `preliminary` job and
  in TheRock's installed-artifact lane, subject to the conditions on each. See
  [Where these tests actually run](#where-these-tests-actually-run).
- **Performance benchmarking.** The lane that benchmarks this library runs nightly, on gfx950,
  outside this repository. See
  [Performance and Benchmarking Testing](#performance-and-benchmarking-testing).

### Supported Configurations

Where hipBLASLt tests actually run. This table describes validation of *this component's tests*, not
the set of architectures the library supports or builds for.

| Configuration | Validation level | Frequency | Notes |
| --- | --- | --- | --- |
| Linux, gfx90a / gfx942 / gfx950 / gfx12 | Client test suite | PR, via Math CI `precheckin` | Broadest per-PR hardware coverage, and the only per-PR signal on gfx950. Also compiles for gfx1250 |
| Linux, gfx90a / gfx942 / gfx950 / gfx12 | TensileLite `common` suite, then the whole unit and characterization tree | PR, via Math CI `preliminary` | Skipped unless `tensilelite/` changed. The second stage also needs the first to pass and a `develop` target |
| Linux, gfx94X-dcgpu | Full | PR (quick), nightly (comprehensive), via TheRock | Primary GitHub Actions test target. 6 shards |
| Linux, gfx90a | Partial | PR (HOST_ASAN quick tier), nightly | Sanitizer lane's default architecture |
| Linux, gfx950-dcgpu | Full | Postsubmit and nightly in the TheRock lane, **not PR** there | Runner capacity, ROCm/TheRock#3288. Covered per PR by Math CI |
| Windows, gfx110X | Partial | PR (quick), nightly | 1 shard |
| Windows, gfx1151 | Partial | Nightly | Forced to the quick tier regardless of requested tier, for memory reasons |
| Linux, gfx950 (MI350) | Benchmarks only | Nightly, in the rocPTS lane outside this repository | Non-gating, and not surfaced on a pull request. See [Performance and Benchmarking Testing](#performance-and-benchmarking-testing) |
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

## Static Analysis

This section is short on achievements and long on gaps, which is itself the finding. **hipBLASLt has
essentially no code-quality static analysis, and the gate that sounds like it provides some does
not.**

**What the gating job actually does.** Math CI's `static-analysis` job scans the working tree and the
git log for a list of sensitive words, maintained in the CI system rather than in this repository, and
fails the build on any match. It exists to keep internal names and unreleased information out of a
public repository. That is a worthwhile gate and it is genuinely enforced, but it inspects text rather
than code, and it would not notice a null dereference, a resource leak, or an uninitialized read. The
name invites exactly the wrong conclusion, which is why it is called out here rather than left in a
table row.

**C++ analysis: none.** There is no `.clang-tidy` file anywhere in hipBLASLt, and although
`cmake/modules/ClangTidy.cmake` exists at the repository root, hipBLASLt's build never references it.
Neither does it use the `CppCheck.cmake` that some sibling projects in this repository wire up. The
top-level `CMakeLists.txt` sets no warning escalation of its own. So the C++ library, which is the
bulk of the shipped product, receives no static analysis on a pull request from any source.

**CodeQL exists but not where it would help.** Two separate things carry the name. The GitHub Actions
workflow [`codeql.yml`](../../.github/workflows/codeql.yml) runs on a weekly schedule and analyzes
only the `python` and `actions` languages, so it never runs on a pull request and never sees C++.
Math CI also defines a `codeql` job for hipBLASLt that does a real compile, but it is not in the
gating set. Between them, C++ CodeQL coverage on a pull request is zero.

**Python analysis is configured down to almost nothing, and unenforced.** `tox -e lint` runs flake8
over `Tensile`, but the `[flake8]` section in
[`tensilelite/tox.ini`](tensilelite/tox.ini) sets `ignore = E, W`, which disables every pycodestyle
error and warning and leaves only the pyflakes checks: undefined names, unused imports, redefinitions.
Those are worth having and they catch real bugs. But the `max-line-length = 132` sitting above that
line has no effect, since line length is an `E` code. More importantly, no CI job runs this
environment, so the check is available rather than applied.

There is no type checking at all, despite type hints on function signatures being a documented style
rule. A `mypy` or `pyright` run would be the highest-value addition here, because TensileLite is a
large, dynamically typed code generator.

**What does run** is formatting, not analysis, and it should not be counted as the latter.
`pre-commit` enforces `clang-format` across C and C++ sources and `black` across Python, and both are
gating. Math CI's static analysis class contains a clang-format checking routine as well, but the job
disables it, so formatting enforcement comes from `pre-commit` alone.

The practical consequence is that defects which a standard analyzer would catch cheaply, at authoring
time, are instead left to code review, the sanitizer lane, and the test suite. The
[roadmap](#improvement-roadmap) proposes closing this, starting with the Python side, which is the
cheapest by a wide margin.

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
  [Supported Configurations](#supported-configurations) table.

## Coverage Expectations by Change Type

- **New hardware-independent logic (TensileLite Python).** A real unit test asserting intended
  behavior, keeping the whole-project floor and every per-file floor intact. A characterization golden
  is not a substitute for new code: pinning behavior you just wrote pins whatever you happened to
  write.
- **Refactoring TensileLite behind existing goldens.** Goldens unchanged is the pass condition. If the
  refactor makes a module unit-testable, converting its characterization coverage to unit tests in the
  same PR is the highest-value thing you can do, and the split card is where to show it.
- **New hardware-independent logic (C++).** A unit test if the code is reachable from
  `hipblaslt-test`. If it is not, say so in the PR and prefer a structure that is.
- **New on-device behavior.** A client test case with numerical validation against the CPU reference.
- **New public API.** An API or descriptor test in the auxiliary suite.
- **Bug fix.** A regression test that fails without the fix. If the fix is not yet available, land the
  reproducer quarantined in `known_bugs.yaml` with a removal note, following the existing pattern.
- **Change to TensileLite behavior.** Update the affected golden on the smallest node id, and explain
  the diff in the PR description.
- **Performance-sensitive path.** Benchmark it yourself and put the before-and-after numbers in the
  description. No lane will do it for you, and nothing fails on a regression, so the description is
  the only control there is.
- **New GPU or OS support.** Validation on that configuration, plus an update to the
  [Supported Configurations](#supported-configurations) table.
- **Packaging or build change.** Install and layout validation; the pre-flight layout check must pass.

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
forces rather than one anybody chose. The churn and complexity that make it necessary, and the phased
plan for making it unnecessary, are in
[why the net came before the tests](#how-this-one-was-learned-why-the-net-came-before-the-tests). The
cost of that order is a coverage number that looks better than the underlying quality warrants, which
is why this document keeps separating the two rather than quoting the union and moving on.

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
correctness load; unit testing carries the generator; performance testing carries nothing that ever
reaches a change. The first two are appropriate. The third is a genuine weakness for a library whose
entire reason for existing is throughput.

## Key Quality Concerns

The things that must not break, in rough order of how much damage a miss does.

**1. Numerical correctness.** A GEMM that returns wrong numbers, silently, is the worst failure this
library can produce, because it propagates into model outputs with no error anywhere. It is
validated by the client suite comparing against a CPU reference with tolerance checks across the
datatype and shape space, plus dedicated ULP-level tests. Silent-wrong-answer bugs are exactly what
the reproducer-then-quarantine pattern in `known_bugs.yaml` is designed to prevent from recurring.

**2. Performance.** Users choose hipBLASLt for throughput; a correctness-preserving change that costs
20% is a real defect. Pull requests are not benchmarked, so that cannot be caught before a merge. A
nightly lane outside this repository measures gfx950 and alerts its owners, so a regression is found
a day or more later, against a range of commits, and by someone other than the author. It is the
largest gap in this document.

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
- **Performance review.** Nightly data exists in rocPTS for gfx950, and nothing else is measured. The
  review of it is manual and informal. This is the release step most in need of automation.
- **QA handoff.** System-level and release validation are executed by QA outside this repository.

## Dependencies and Validation Handoffs

Where confidence comes from and where ownership changes hands.

| Dependency | Owning team | How it is validated | Known gap |
| --- | --- | --- | --- |
| TheRock (build and shared test infrastructure) | TheRock team | PR, nightly, and release lanes | Runner capacity limits which architectures run per PR |
| Math CI (client, TensileLite and coverage jobs) | DevOps / Math CI | Per-PR checks on the same PR | Configuration lives in an AMD-internal repository, so contributors cannot see what runs or which jobs gate |
| Nightly performance measurement (`rocPTS`) | Performance tracking team | Nightly benchmarks of a monorepo build on gfx950 | Lives outside this repository and reports to its own owners, so no result reaches a pull request or names a commit |
| HIP, ROCr, compiler toolchain | Core ROCm teams | Consumed via TheRock; validated by their own CI | A toolchain regression surfaces here as a hipBLASLt test failure, and triage cost lands on this team |
| `rocisa` | Component team (in this repository) | Own tox environment and CI lane | |
| Downstream frameworks | Framework teams | Integration testing outside this repository | No pre-merge signal; regressions are found after the fact |
| QA validation | QA | Release qualification | |

The honest summary is that hipBLASLt depends on two CI systems it does not own, and the boundary
between them is not documented anywhere a contributor can see. That is the most common source of
"I thought CI covered that" confusion in this component.

## Improvement Roadmap

Ordered by value per unit of effort, not by ambition.

### Near term, cheap and unblocking

1. **Fix the tier filter in the TheRock test driver** so tiers above `quick` select their categories
   instead of running the whole binary. One-line fix; makes the documented taxonomy real.
2. **Make the installed-artifact lane's snapshot behavior deliberate.** Either ship syrupy with the
   installed test tree so the goldens are checked there too, or state in the lane that snapshot
   coverage is intentionally left to the source lanes. Today it is a silent skip that reads like an
   accident.
3. **Publish the required-check list.** Document which checks actually block a merge, so contributors
   and reviewers stop guessing. The four-lane table under
   [Where these tests actually run](#where-these-tests-actually-run) is a start, but it belongs
   somewhere it cannot drift from the branch-protection settings themselves.
4. **Make the test suite resolve its own toolchain.** Characterization tests locate `amdclang++`
   through a bare `shutil.which`, so whether they pass depends on how the surrounding lane happened
   to order `PATH`. The same tests then behave differently in different lanes for reasons that have
   nothing to do with the code under test. Resolving the toolchain inside the tox environment removes
   a recurring source of false failures.
5. **Enforce `--strict-known-bugs` in its own lane** (AIHPBLAS-4196). The detection already exists;
   what is missing is a job that fails on a stale entry. A dedicated GitHub Actions job is the right
   home for it, because the flag cannot be turned on inside the build without failing local developer
   builds. Worth extending to orphaned entries, which are silently ignored today.
6. **Govern known-bug entries as one thing.** Seven mechanisms suppress or record known-bad behavior
   and none of them share a convention. The proposal, which needs team agreement before it becomes
   policy, is four rules:

   - Every entry names its ticket in a machine-readable field rather than a comment, so ownership and
     status live on the ticket where they can be kept current, instead of as a name in a file that
     goes stale.
   - Every entry carries a review date.
   - The suppressed code keeps running wherever the mechanism allows, so a fix can be observed rather
     than assumed.
   - A fix fails the build, so the entry deletes itself.

   Two mechanisms cannot satisfy the third rule today, so adopting this means changing them or
   accepting a stated exception. Start by splitting flaky from known-failing in the client quarantine
   list, which is the largest and blindest surface.

7. **Run the Python linter that already exists.** `tox -e lint` is configured and invoked by nothing.
   Wiring it into the existing TensileLite GitHub Actions lane is close to free, and widening it past
   `ignore = E, W` can be done gradually. Rename or re-describe the `static-analysis` job at the same
   time, since its current name causes people to believe code analysis is already covered.

### Medium term, the structural unlock

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
5. **Fold the data-consistency checks in `test_PlaceholderMerge.py` and `test_GpuRevisionTarget.py`
   into `TensileLogic --check-all`, with one asymmetry to design around first.** Both files validate
   the same class of thing that checker already owns: logic YAML data, no code, no GPU. Today they
   only run inside the pytest suite, gated on the raw corpus being on disk (`_needs_logic_dir`), which
   is permanently false in TheRock CI's installed-artifact layout and conditionally skipped in Math CI
   on YAML-only diffs (see [CI visibility and gating](#ci-visibility-and-gating)). `TensileLogic
   --check-all` runs unconditionally wherever kernels are generated, so moving a check there closes
   the "corpus is on disk" gap unconditionally, and closes the "runs on every PR" gap for whatever
   architectures that PR's builds actually target.

   TheRock's gfx1250 lane makes the case concretely. gfx1250 has its own gap on top of
   `_needs_logic_dir`: TheRock's test runner
   (`build_tools/github_actions/test_executable_scripts/test_tensilelite.py` in the TheRock repository)
   skips the entire `Tensile/Tests/unit` directory for this family
   (`UNIT_TEST_SKIP_FAMILIES = {"gfx1250"}`, because it runs under GPU emulation with no arch-specific
   unit tests), running only the `Tests/common` GEMM suite instead. That is the exact architecture the
   2026-08-27 break was on, and it is a second, independent reason `test_PlaceholderMerge.py` would
   never have run there even if the raw corpus had been present. `TensileLogic --check-all` runs as a
   CMake build step, not through this test runner, so it is unaffected by `UNIT_TEST_SKIP_FAMILIES`;
   TheRock does build a gfx1250 target (`gfx125X-dcgpu` in TheRock's `amdgpu_family_matrix.py`,
   build-only today pending hardware), so moving the check there would have exercised gfx1250's logic
   files in TheRock CI specifically, not just in principle.

   That "closes the gap for whatever architectures that PR's builds actually target" clause is the
   asymmetry. Sibling-`DeviceNames` consistency compares files within one
   `(codename, arch, basename)` group, so it moves cleanly: the comparison never needed files outside
   the architecture being built, and it survives the per-arch filtering that
   [PR #9218](https://github.com/ROCm/rocm-libraries/pull/9218) put on the build-wired `--check-all`
   invocation (see [above](#why-it-runs-in-the-build)). It does need `check-all`'s per-file worker loop
   extended with a cross-file grouping pass, which it does not have today; every existing validator
   there looks at one file in isolation.

   The chip-ID-aware-arch lock does not move as cleanly. It is parametrized over every architecture in
   the tree specifically to assert a whole-corpus fact: that only gfx950 carries chip-ID-aware
   predicates. Folding it into the build-wired invocation as-is would make it silently check only
   whichever architectures a given build happens to target, which is a regression from what it
   guarantees today: a single-arch CI build would "pass" a check whose entire job is to notice a
   second architecture picking up chip-ID logic it should not have. Preserving that guarantee means
   either invoking this one check with an explicit `--architecture all` regardless of the build's own
   `GPU_TARGETS`, or leaving it as a pytest test that always sees the full corpus.

   Either way, moving what does move trades the pytest tests' per-node test report for the existing
   checker's build-blocking, unnamed-check failure mode, and needs
   `TensileLogic/known_bugs.yaml`'s schema extended to key on a basename or file pair rather than one
   path plus `SolutionNameMin`, since a sibling mismatch is inherently about two files. The other three
   tests in `test_PlaceholderMerge.py`, an AST scan of `SolutionLibrary.py` plus two function-level
   unit tests, validate code rather than data and should stay pytest tests.

### Longer term, the real gap

1. **Get a performance number onto the change.** Pull requests are not benchmarked today, so this
   builds something new rather than repairing something. The nightly lane already has the pieces
   worth reusing: version-controlled datasets, a comparison against a baseline, and a runner pool.
   What we would like, roughly in order:

   - A measurement tied to one commit, on a small and fast problem set, with the result visible to
     the author on the pull request. Feedback first; gating only once it is stable enough not to
     become the thing everyone reruns.
   - Architecture coverage beyond gfx950, prioritized by where the library ships tuned kernels and
     where customers run.
   - Shape coverage along the two axes today's sample holds fixed: batch counts, including batched
     FP8, and padded leading dimensions.
   - A stability story before a threshold: enable the harness's adaptive sampling, or pin clocks, or
     both, so the noise floor is a measured quantity rather than an assumption.
   - Problem lists version controlled in this repository, so a change to what we measure is reviewed
     like any other change. This is how the nightly lane manages its datasets and is worth copying.
   - A lane that reports its own failure. One that says nothing when it works and nothing when it
     breaks will stop working without anyone noticing.
   - A named owner for the signal.

2. **Graduate mutation testing** from a report-only pilot to a maintained signal on the modules where
   it has demonstrated value. This is the companion to the migration item above: until a module has
   real unit tests, its mutation score is the only evidence that its goldens would actually catch a
   regression rather than just execute the code.
3. **Prune redundant numerical variants** in the client suite to buy back runtime, and spend it on
   architecture breadth per PR.
4. **Introduce real static analysis on the C++ library.** Add a `.clang-tidy` with a deliberately
   small starting rule set, wire it into the build the way sibling projects in this repository
   already do, and ratchet it rather than trying to land a clean full-strength run. Adding `cpp` to
   the CodeQL language matrix is the cheaper first step and would at least produce a weekly signal.
   Type checking on TensileLite belongs in the same effort.

## Known Risks and Gaps

Stated plainly, so none of these are a surprise at release time. Twenty-seven gaps, grouped by theme.

**On the Tracking column.** This table names no owner. Per-gap ownership changes far more often than
this document does, and a stale name in a repository file is worse than no name because it reads as
authoritative. Instead each row points at a work item, which is where ownership, status and priority
can actually be kept current. An empty cell is meaningful: it means the gap is real, acknowledged,
and not yet tracked anywhere. Most of them are empty right now, and closing that is the first thing
this table should drive.

### Performance

| Gap | Regression risk | Impact | Mitigation today | Tracking |
| --- | --- | --- | --- | --- |
| Pull requests are not benchmarked, so a regression cannot be attributed to the change that introduced it | High | High | A nightly lane measures a monorepo build, which narrows a regression to a day of merges rather than to a commit |  |
| No performance gate, and no result surfaced in this repository | High | High | The nightly compares against a baseline and alerts its owners, so catching a regression depends on someone outside this repository reading it |  |
| Benchmark coverage is one architecture family, gfx950 | Medium | Medium | Correctness coverage is broader; performance on the other architectures the library tunes for is not measured on a cadence |  |

### Coverage and verification

| Gap | Regression risk | Impact | Mitigation today | Tracking |
| --- | --- | --- | --- | --- |
| Enforced coverage counts characterization scaffolding as unit testing, overstating how much is verified | High | Medium | The split summary card reports the characterization-only share, but it gates nothing |  |
| Mutation testing, the only evidence a golden would catch a regression, covers eight files and runs nowhere in CI | Medium | Medium | Manual `tox -e mutation-unit`; widening PRs are in draft. Treated as report-only | AIHPBLAS-3868 |

### CI visibility and gating

| Gap | Regression risk | Impact | Mitigation today | Tracking |
| --- | --- | --- | --- | --- |
| What Math CI runs, and which of its jobs gate, is not visible from this repository | Medium | Medium | This document, which is a snapshot and will drift |  |
| Which checks are actually required to merge is undocumented | Medium | Medium | Institutional knowledge |  |
| `preliminary` is dropped from the gating list when a PR also touches rocroller | Medium | High if hit | TheRock's lane still runs the unit tree, but the four-architecture GPU coverage is lost silently |  |
| The `preliminary` stage that runs the unit and characterization tree is conditional on the target branch | Medium | Medium | Most pull requests target `develop` and do get the full gate |  |
| Every lane is fail-fast, so one early failure erases the signal from every stage behind it | Medium | Medium | Reading a red run means checking what did not get to run, not just what failed |  |
| The same ~6,000 TensileLite tests run in four lanes, three holding a GPU only one of them needs | Low | Low | Expensive in runner capacity; the redundancy does buy independent confirmation |  |
| The installed-artifact lane silently skips the snapshot tests, since syrupy is not in the installed tree | Low | Low | The goldens are enforced upstream; the skip is stated in `conftest.py` but reads like an accident |  |
| Tiers above `quick` apply no filter in the TheRock lane, so the taxonomy is half-real | Medium | Medium | CTest honors the tiers correctly when used |  |
| Submodule-bump pull requests run a reduced test set relative to source changes | Medium | Medium | Owned outside this component; noted because failures have been merged past |  |
| Math CI's `preliminary` job appears to skip the `Tensile/Tests/unit` suite entirely on YAML-only diffs, running only numeric/solution-correctness checks instead | High | High if hit | None observed. Confirmed by the 2026-08-27 `develop` break: a 444-file, YAML-only PR (#11274) never ran the suite containing the sibling-`DeviceNames` check, and the resulting data bug only surfaced on a later, unrelated PR that happened to touch `.py` files |  |
| The `_needs_logic_dir` xfail (see [Known Bugs and Expected Failures](#known-bugs-and-expected-failures)) is unconditional in TheRock CI, so the logic-corpus consistency checks it guards never execute there | Medium | High if hit | Math CI can still catch it when its own suite actually runs, but see the row above for when it does not | |
| For gfx1250 specifically, TheRock's `Tensile/Tests/unit` directory (which holds both logic-corpus consistency test files) is skipped outright by the test runner, independent of `_needs_logic_dir`: `test_tensilelite.py`'s `UNIT_TEST_SKIP_FAMILIES = {"gfx1250"}` skips the entire unit-test directory under GPU emulation, running only `Tests/common` instead | Medium | High if hit | None; this is the exact architecture the 2026-08-27 break was on | |

### Known bugs and flaky tests

| Gap | Regression risk | Impact | Mitigation today | Tracking |
| --- | --- | --- | --- | --- |
| No flaky-test tagging or expiry convention, and flaky tests share one list with known-failing ones | Medium | Medium | `known_bugs.yaml` quarantine with ticket references and removal notes |  |
| Known-bad behavior is suppressed in seven places with no shared convention, and the largest excludes the test entirely | Medium | Medium | Per-mechanism discipline is good in places and absent in others; nothing reports the total |  |
| A stale `TensileLogic` known-bugs entry only warns, because `--strict-known-bugs` defaults off | Low | Low | The checker re-validates and reports stale entries on every build | AIHPBLAS-4196 |
| The library logic gate has no check name, no test report, and cannot be run in isolation by CI | Low | Medium | It runs on every kernel-generating build including local ones, which gives it good reach | AIHPBLAS-4196 |

### Static analysis and type checking

| Gap | Regression risk | Impact | Mitigation today | Tracking |
| --- | --- | --- | --- | --- |
| No code-quality static analysis on the C++ library at all: no clang-tidy or cppcheck, and CodeQL does not cover C++ | High | Medium | Code review, the sanitizer lane, and the test suite absorb what an analyzer would catch earlier |  |
| The gating job named `static-analysis` is a sensitive-word scan, so the gate list reads as though code analysis is covered | Medium | Medium | Documented here; the scan does its actual job well |  |
| Python linting is configured (`tox -e lint`) but no CI job runs it, and `ignore = E, W` narrows it to pyflakes | Medium | Low | `black` is enforced through `pre-commit`; pyflakes-class bugs are otherwise caught in review |  |
| No type checking on TensileLite, despite type hints being a documented style rule | Medium | Medium | None. A large dynamically typed code generator with no type verification |  |

### Test surface gaps

| Gap | Regression risk | Impact | Mitigation today | Tracking |
| --- | --- | --- | --- | --- |
| Very little of the C++ library is unit-testable; the blockers are structural | Medium | Medium | Heavy integration coverage compensates, at the cost of slow feedback |  |
| TSAN build options exist but no CI lane uses them, and there is no UBSAN at all | Low | High if hit | None. Thread-safety bugs would be found downstream |  |
| No multi-GPU tests | Low | High if hit | None in this repository |  |
| Three parallel enum-to-string tables can drift | Low | Medium | None automated |  |
| TheRock's lane skips the unit tests when `AMDGPU_FAMILIES` includes `gfx1250`, and gfx950-dcgpu is excluded from PR CI | Low | Medium | Math CI's `preliminary` covers gfx950 per PR; the gfx1250 skip is emulation-mode only | [TheRock#3288](https://github.com/ROCm/TheRock/issues/3288) |

## Owners and Review Cadence

**Document owner:** T.J. Alumbaugh (@talumbau). **Technical lead:** Tony Davis (@tony-davis),
responsible for keeping this document accurate.

**Test ownership** is currently distributed rather than assigned. The client GTest suite, the
TensileLite Python suites, the characterization goldens, and the CI lanes each have people who work
on them, but none has a designated owner. That is a real gap, because a flaky-test policy needs
someone for a flaky test to belong to. Note the distinction this document draws: area ownership is
worth naming here because it changes slowly, while ownership of an individual test, gap, or
quarantine entry belongs on a ticket, where it can be reassigned without editing a file.

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
