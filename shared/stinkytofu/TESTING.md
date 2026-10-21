# StinkyTofu Testing Strategy

- **Owner:** TBD, to be agreed with `@ROCm/stinkytofu-reviewers`
- **Drafted by:** Tony Davis (@tony-davis), as part of the ROCm-wide TESTING.md initiative

> **Status: draft for comment.** This is a first pass written from outside the StinkyTofu team, by
> reading the tree, the CI definitions and the pull request history. It is meant to be corrected by
> the people who build StinkyTofu every day, and every claim in it is open to challenge. Where
> something is inferred rather than confirmed, it says so.
>
> **In a hurry?** Start with [The short version](#the-short-version). From there, jump to
> [Choosing the Right Test Type](#choosing-the-right-test-type) if you are about to write a test, or
> to [Known Risks and Gaps](#known-risks-and-gaps) if you are deciding how much to trust a green
> check.

This document describes how StinkyTofu is tested today, which signals gate a merge, and where the
holes are. It follows the ROCm-wide TESTING.md template, in the same shape as
[hipBLASLt's](../../projects/hipblaslt/TESTING.md) and
[TensileLite's](../../projects/hipblaslt/tensilelite/TESTING.md), so a reader who knows one can
find their way around the others. It describes the current state rather than an aspirational one.

**Scope.** This covers StinkyTofu itself (`shared/stinkytofu/`), its Python bindings, and the
GitHub Actions lanes that run its tests or the tests of its direct consumers. It does not repeat
what the hipBLASLt and TensileLite documents already say about their own suites, and it leaves out
Math CI, which those two documents cover.

One habit carries over from the hipBLASLt document. Where a test practice exists because of a
specific failure, the failure is written up under a heading **How this one was learned**, in a
highlighted box with the finding in one bold sentence and the narrative folded behind a collapsed
**The full account** toggle. A safeguard whose reason is recorded nowhere tends to get removed by
whoever touches it next, and increasingly that is a coding agent. The write-up here is blameless.
It is about what the tests could and could not see, not about who wrote what.

## The short version

**StinkyTofu is designed to be testable, and it is tested.** It is built like a small LLVM: a pass
manager, a textual IR with a parser and printer, a standalone driver (`stinkytofu-opt`), and a
FileCheck-style runner. Nothing in it needs a GPU. That design pays off in a large, fast,
hardware-independent suite of GTest unit tests and pass-level FileCheck tests, run on Linux, under
AddressSanitizer, and on Windows for every pull request that touches it. Unit testing is the right
tool here, and it is the tool already in use.

**What gates a merge.** `Component CI: StinkyTofu` (GCC and amdclang builds, the full CTest suite,
an AddressSanitizer build and test, Windows shared and static builds, and a pip-install test) and
`Component CI: rocISA`, which rebuilds rocisa against StinkyTofu and runs its tests. Both roll up
into `Component CI Summary`, a required check on `develop`. `pre-commit` also gates, and Multi-Arch
CI selects TensileLite's suite for any StinkyTofu change.

**What runs but does not gate.** The clang-tidy lane, and a per-PR gfx1250-strict hardware lane
that builds hipBLASLt and TensileLite and runs their GEMM suites on real hardware. Neither appears
in the required-check list for `develop` as of 19 October 2026.

**The biggest gap is an independent check of the hardware model.** The tests are very good at
answering "did this pass do what its author intended?" Nothing in this directory answers "is what
the author intended what the hardware requires?" The instruction tables, hazard distances and wait
rules are written by hand, and the tests encode the same understanding as the code. The one place
StinkyTofu's output runs on hardware is downstream, does not gate, samples a set of GEMM shapes, and
is unlikely to catch a hazard that fires once in a few thousand launches.

**The second gap is configuration.** Most tests run one pass at a time with that pass's standalone
defaults, and those defaults are not always the production configuration. The clearest case is
`InsertWaitAluPass`, the pass that places every hazard wait in gfx1250 expert scheduling mode 2: no
test in this directory runs it with the exact option set the gfx1250 backend uses.

**How to use this document.** It is a living artifact, meant for regression analysis (what was
supposed to catch this?), release-readiness review, onboarding, and planning CI work. Every gap
below is a place where "we assumed CI was covering it" would otherwise get said later.

## Contents

The sections marked **P1** are the ones the ROCm template treats as highest priority. The short
version above and this list are additions to the template; everything else follows it.

**What we test, and how**

- [Component Overview](#component-overview) **P1**
- [Development Workflow](#development-workflow) **P1**
- [Testing Strategy and Layers](#testing-strategy-and-layers) **P1**
  - [Unit Testing Strategy](#unit-testing-strategy)
    - [Is the design testable, and is it tested?](#is-the-design-testable-and-is-it-tested)
    - [What the tests run versus what production runs](#what-the-tests-run-versus-what-production-runs)
  - [Pass-Level FileCheck Tests](#pass-level-filecheck-tests)
  - [Python Binding, API and Build-Time Tests](#python-binding-api-and-build-time-tests)
  - [Downstream Integration Testing](#downstream-integration-testing)
  - [Performance and Benchmarking Testing](#performance-and-benchmarking-testing) **P1**
  - [How this one was learned: a gate that turned a protection off](#how-this-one-was-learned-a-gate-that-turned-a-protection-off)
- [Pre-submit / CI Gates](#pre-submit--ci-gates) **P1**
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

StinkyTofu is a pass-based optimizer for AMD GPU assembly kernels. TensileLite, the Python code
generator behind hipBLASLt's GEMM kernels, hands it a kernel; StinkyTofu schedules the
instructions, inserts the waits and hazard protections the hardware needs, manages register-mode
state, and emits assembly. Today it targets gfx1250 (including the gfx1250-strict variant), and
TensileLite uses it for the gfx1250 kernels it generates for hipBLASLt and hipSPARSELt. See
[README.md](README.md) and [docs/developer/architecture.md](docs/developer/architecture.md) for the
design.

It has two IR levels. A Logical IR is architecture-independent; an Asm IR is concrete per
architecture, and can be lowered to from Logical IR or parsed directly from assembly text. Passes
run under a pass manager modeled on LLVM's, and the per-architecture hardware model (instruction
definitions, formats, costs) is generated from `.def` files by an in-tree TableGen-like tool.

**One role shapes almost everything below.** On gfx1250, StinkyTofu can switch a kernel into
expert scheduling mode 2 (ESM2, `HW_REG_WAVE_SCHED_MODE=2`). In that mode the hardware stops
checking some register hazards itself, for example a matrix instruction still reading a register
that a later load is about to overwrite. StinkyTofu's `InsertWaitAluPass` must then place every
`s_wait_alu` that protects those hazards. A missing wait does not crash anything. It produces a
wrong answer, rarely and depending on timing. That makes the hazard model the most
correctness-critical part of the component, and the hardest to test.

**How it reaches its consumers.** TensileLite's `rocisa` extension (`_rocisa`) links
`stinkytofu::stinkytofu`, and the rocisa-to-StinkyTofu conversion code under
[`src/conversion/rocisa/`](src/conversion/rocisa/) is compiled directly into `_rocisa` rather than
into StinkyTofu's own library. A separate Python package,
[`rocisa_stinkytofu_adaptor`](../../projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/),
provides a rocisa-shaped interface backed by StinkyTofu's Logical IR. See
[TensileLite's rocisa section](../../projects/hipblaslt/tensilelite/TESTING.md#rocisa) for that
side of the coupling.

Major dependencies: the ROCm compiler toolchain, `amd_comgr` (used on the CPU as an assembler and
an instruction-support probe, never to run anything), GTest, nanobind for the Python bindings, and
the shared build and CI infrastructure provided by TheRock.

## Development Workflow

What a developer does between making a change and getting it merged.

**1. Build.** `invoke build` from `shared/stinkytofu` is the standard loop (see
[README.md](README.md) and [tasks.py](tasks.py)). It finds ROCm, configures with CMake, builds the
library, tools, tests and Python bindings, and always passes `-DSTINKYTOFU_ENABLE_WERROR=ON`. When
StinkyTofu is built as a sub-project of hipBLASLt, its tests and Python bindings are off by default.

**2. Run the tests that match what you touched.** None of these need a GPU.

| You changed | Run this |
| --- | --- |
| Anything in the library or tools | `cd build && ctest` (unit, API, FileCheck, and Python tests) |
| One area, iterating | `./build/tests/unit_tests --gtest_filter="DAGSchedulerPassTest.*"`, or `ctest -R FileCheck` |
| Memory handling | `invoke asan`, then `ctest -LE python` in `build_asan` |
| Anything, before pushing | `invoke tidy` (after `invoke build`) and `pre-commit run --files <changed files>` |
| Line coverage, locally | `invoke coverage`, which writes an HTML report and an `lcov` file |
| The rocisa conversion code or the adaptor | `cd projects/hipblaslt/tensilelite && tox -e rocisa`, and the adaptor's own `rocisa_stinkytofu_adaptor/tests/test.sh` |
| Behavior TensileLite depends on | The TensileLite suites; see [its development workflow](../../projects/hipblaslt/tensilelite/TESTING.md#unit-testing-strategy) |

Most developer documents under [docs/developer/](docs/developer/) and [docs/user/](docs/user/) name
the specific unit test and FileCheck file for the feature they describe, with a reproduction
command. That is an unusually good habit, and the place to look first.

**3. Add the right kind of test.** See [Choosing the Right Test Type](#choosing-the-right-test-type).
For a bug fix, the expectation is a regression test that fails before the fix and passes after it.

**4. Open the PR** against `develop`, with a Conventional Commits title (`fix(stinkytofu): ...`) and
a JIRA or issue reference line.

**5. Watch the right checks.** See [Pre-submit / CI Gates](#pre-submit--ci-gates). The fastest
useful signal is `Component CI: StinkyTofu`; the gfx1250-strict hardware lane takes much longer.

## Testing Strategy and Layers

Five layers carry the load. **Unit tests** (GTest) check data structures, analyses and individual
passes from C++. **Pass-level FileCheck tests** feed a small assembly or IR input through one or a
few passes with `stinkytofu-opt` and assert on specific output lines. **Python binding, API and
build-time tests** check what consumers link against and what the build generates.
**Downstream integration** is where StinkyTofu's output meets rocisa, TensileLite and, in one lane,
real gfx1250 hardware. **Performance testing** does not exist as a lane; see
[Performance and Benchmarking Testing](#performance-and-benchmarking-testing).

### Unit Testing Strategy

**Purpose.** Validate StinkyTofu's logic without hardware: IR construction and parsing, analyses,
register allocation, scheduling, wait insertion, hazard passes, and error handling.

| Item | Detail |
| --- | --- |
| Framework | GTest, through CTest (`gtest_discover_tests`, label `unit_tests`, 60 s timeout per test) |
| Location | [`tests/unit/`](tests/unit/), in subdirectories that mirror `src/`: `asm/`, `ra/`, `ssa/`, `logical/`, `core/`, `ir/` |
| Binary | `unit_tests`, linked against `stinkytofu_static` so tests can reach symbols the shared library does not export |
| How to run | `cd build && ctest -L unit_tests`, or the binary directly with `--gtest_filter` |
| Suite size | Well over a thousand test cases, most of them in `tests/unit/asm/` |

#### Is the design testable, and is it tested?

These are the two questions the TESTING.md initiative most wants answered for each component, so
this subsection answers them directly.

**Is the design testable? Yes, and unusually so for a ROCm component.** The seams a unit test needs
are already there, by design rather than by retrofit:

- **Passes are IR in, IR out.** A pass is a class with a `run(Function&, PassContext&,
  AnalysisManager&)` method under a pass manager
  ([`include/stinkytofu/core/PassManager.hpp`](include/stinkytofu/core/PassManager.hpp)). A test
  builds or parses a function, runs one pass, and inspects the result.
- **The IR has a text form, with a parser and a printer.** That is what makes FileCheck tests
  possible at all, and it means a reduced reproducer from a real kernel can become a test input
  without writing C++.
- **The hardware model is data.** Instruction definitions, formats and costs live in `.def` files
  under [`hardware/`](hardware/) and are turned into tables by [`tools/tablegen/`](tools/tablegen/).
- **No GPU, no HIP runtime.** The library calls no GPU runtime API. `amd_comgr` is used only to
  assemble instructions on the CPU and to probe which instructions a toolchain supports.
- **White-box access is deliberate.** The test binary links a static copy of the library, so
  internal helpers can be tested directly; a separate `api_tests` binary links the shared library to
  catch a symbol a consumer needs but the library forgot to export.
- **Little hidden global state.** There are two environment-variable reads in the whole library
  (an intrinsic search path override and a debug toggle) and a handful of read-mostly registries.

The one part that is hard to unit test is the rocisa conversion code in
[`src/conversion/rocisa/`](src/conversion/rocisa/). It takes live rocisa objects exposed through
nanobind and turns them into StinkyTofu instructions, so testing it in C++ needs either the real
`_rocisa` extension or hand-built fakes of rocisa's container types, and neither exists today.

**Is it tested? Mostly, yes.** The suite is large and growing, it runs in minutes on an ordinary
build machine, and most pull requests that change behavior also change tests. The tests assert
intended behavior, written down by the author: the FileCheck fixtures make targeted assertions on
specific output lines, and none of them compares a whole output file against a saved copy. The
right investment from here is more targeted unit and FileCheck tests where they are thin, plus the
independent checks described under [Improvement Roadmap](#improvement-roadmap).

**Where the unit tests are thin.** The following come from matching source files against test
files and test inputs by name, so read them as pointers to check rather than as proof that code is
untested. Much of what is listed is exercised indirectly through other tests.

- `InsertWaitAluPass`, the largest of the safety-critical passes, has no GTest file of its own. Its
  behavior is checked through a small number of FileCheck fixtures.
- The wait-count dataflow under [`src/transforms/asm/waitcnt/`](src/transforms/asm/waitcnt/)
  (`WaitDataflow`, `ShallowPredPromotion`, `WmmaRunWaitMerge`) has no direct unit test; it is
  reached through the `StinkyWaitCntInsertionPass` unit tests and fixtures.
- Several analyses have no direct test: `HazardGapAnalysisPass`, `Dominance`, `SSAFunctionShape`
  and `SSASlotIndexes`. The DAG data structure `RegionDAG` and the register-lifting code in
  `ssa/AsmRegisterLifting` are reached only through the passes that use them.
- The rocisa conversion code has no test in this directory (see above and
  [Downstream Integration Testing](#downstream-integration-testing)).
- `Gfx1250Backend`, which assembles the production pipeline, is tested for configuration checks and
  abort behavior (`tests/unit/core/BackendTest.cpp`), but not for which passes run, in what order,
  with which options.

**Explicitly not covered by unit tests:** whether the hardware model matches the hardware, whether a
scheduled kernel computes the same result as the unscheduled one, and anything timing-dependent.
Those need either execution or an independent checker, and are discussed under
[Why We Test This Way](#why-we-test-this-way).

#### What the tests run versus what production runs

Most tests, unit and FileCheck alike, run one pass or a short list of passes with each pass's
default options. Production runs the full pipeline that
[`src/pipeline/backend/Gfx1250Backend.cpp`](src/pipeline/backend/Gfx1250Backend.cpp) builds, with
options chosen by the backend and by TensileLite's module options. Where the two differ, the
production configuration can go untested even when each individual behavior has a test.

The case that matters most is `InsertWaitAluPass`. Its three options default to off when it runs
standalone ([`InsertWaitAluPass.hpp`](include/stinkytofu/transforms/asm/InsertWaitAluPass.hpp)).
In production, `gfx1250InsertWaitAluOptions` always turns two of them on (`sharedOrderCountFollowers`
and `xdlCountFromNextWmma`, two rules that let a wait carry a larger count) and turns the third on
when the kernel asks for source-operand tracking. As of this writing, each of those rules has its
own FileCheck fixture, but no test in this directory runs the pass with the combination production
uses, and no test sets `EnableESM2` on a module and runs the registered pipeline. Only one FileCheck
fixture uses `stinkytofu-opt`'s `-O<N>` pipeline mode at all. The full production configuration is
exercised only downstream, through rocisa and TensileLite.

This is cheap to close, and it is first in the [roadmap](#improvement-roadmap): a set of fixtures
that run through `-O<N>` with the module options TensileLite actually sets for gfx1250, so the
configuration that ships is the configuration that is tested.

### Pass-Level FileCheck Tests

| Item | Detail |
| --- | --- |
| Runner | [`tools/stinkytofu-check`](tools/stinkytofu-check/stinkytofu-check.cpp), an in-tree FileCheck-style tool supporting `RUN`, `CHECK`, `CHECK-NEXT`, `CHECK-SAME`, `CHECK-NOT`, `CHECK-DAG`, `CHECK-LABEL` and `XFAIL` |
| Driver | `stinkytofu-opt`, which parses the input, runs the passes named on the `RUN` line, and prints the result |
| Location | [`tests/filecheck/`](tests/filecheck/): `.stir` (StinkyTofu IR) and `.s` (raw assembly) inputs, each registered as a CTest case `FileCheck.<name>` |
| Suite size | Over a hundred fixtures, the largest groups covering implicit-dependency building, wait-count insertion and DAG scheduling |

In LLVM's terms these are unit tests of a pass: one small input, one pass or a short pipeline, and
assertions on the lines that matter. They are the cheapest way to turn a reduced reproducer from a
real kernel into a permanent test, and most of the wait and hazard behavior is tested this way.

Two properties are worth knowing. Every fixture passes an explicit `--arch`, and today all of them
target gfx1250, with a single fixture under `gfx1250-strict/` (registered only when that variant is
built). And because `stinkytofu-opt` runs passes with their standalone defaults unless the `RUN`
line says otherwise, a fixture tests the options on its `RUN` line, which may not be the options
production uses (see the previous section).

One behavior of the runner is worth knowing when reading a green result. `stinkytofu-check` decides
pass or fail from the `CHECK` lines alone. If `stinkytofu-opt` exits with an error, the runner
prints a warning and still passes the fixture when the output it did produce matches; and `XFAIL`
notes an expected non-zero exit without requiring one. In practice most fixtures print their output
after all passes run, so an early abort usually fails the `CHECK` lines anyway, but treating an
unexpected exit status as a failure would close the gap.

The tests in `tools/stinkytofu-opt/tests/` are sample inputs for the driver, not registered tests.

### Python Binding, API and Build-Time Tests

- **Python bindings.** pytest files under [`python_module/tests/`](python_module/tests/), registered
  in CTest with the label `python` when the bindings are built and pytest is available. One module
  skips itself when the binding is not built. CI also runs them against a fresh `pip install`.
- **Exported API.** `api_tests` ([`tests/api/`](tests/api/)) links the shared library and checks
  that the symbols rocisa and `stinkytofu-opt` use are exported, along with the plugin-loading
  tests (a good plugin and one with a mismatched version).
- **Link-level guard.** `LibStinkytofuHasNoPluginDependency` runs `readelf` on the built shared
  library to assert it does not depend on a plugin library.
- **Generated tables.** `GenInstructionsTest` checks that the generated instruction tables match the
  configured architecture list.
- **Parser robustness.** [`tests/unit/asm/FuzzTest.cpp`](tests/unit/asm/FuzzTest.cpp) feeds edge-case
  inputs (empty, very long, null bytes, deep nesting, large numbers) to the IR and pattern parsers
  and checks that they report errors rather than crash. These are hand-written cases, not a
  coverage-guided fuzzer.

### Downstream Integration Testing

StinkyTofu's output is consumed through rocisa and TensileLite, so part of its testing happens in
their suites. Four lanes matter.

- **`Component CI: rocISA`** runs whenever StinkyTofu changes, because
  [`component_ci.py`](../../.github/scripts/component_ci.py) maps `shared/stinkytofu/**` to it. It
  pip-installs rocisa and runs [`rocisa/test/`](../../projects/hipblaslt/tensilelite/rocisa/test/),
  including two plugin configurations that turn "plugin not built" skips into failures. Some of
  those tests drive StinkyTofu through rocisa's production path; for example
  [`test_mubuf.py`](../../projects/hipblaslt/tensilelite/rocisa/test/test_mubuf.py) carries the
  regression test from [PR #11153](https://github.com/ROCm/rocm-libraries/pull/11153), described in
  the write-up below.
- **The adaptor's emission-consistency tests.**
  [`test_emission_consistency.py`](../../projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/tests/test_emission_consistency.py)
  builds the same kernel three ways (native rocisa; rocisa converted to StinkyTofu; the adaptor
  through StinkyTofu's Logical IR) and requires byte-identical assembly. It is the only test that
  exercises the conversion code in `src/conversion/rocisa/` from end to end. As far as this
  document can tell, no GitHub Actions workflow runs the adaptor's tests; they run through their own
  `test.sh`.
- **TensileLite through Multi-Arch CI.** TheRock's test selector maps `shared/stinkytofu` to the
  `tensilelite` job, so a StinkyTofu change runs TensileLite's suite; see
  [How Multi-Arch CI decides whether TensileLite runs at all](../../projects/hipblaslt/tensilelite/TESTING.md#how-multi-arch-ci-decides-whether-tensilelite-runs-at-all).
  Seven gfx1250 TensileLite configurations carry a `skip-stinkytofu` mark and are skipped when
  TensileLite uses the adaptor backend (`ROCISA_BACKEND=stinkytofu`).
- **The gfx1250-strict hardware lane.**
  [`gfx1250_hipblaslt_tensilelite_test.yml`](../../.github/workflows/gfx1250_hipblaslt_tensilelite_test.yml)
  runs on every non-draft pull request from a ROCm organization member that touches
  `shared/stinkytofu/` (among other paths), and nightly on `develop`. It builds hipBLASLt and
  TensileLite for gfx1250-strict from the pull request's source, then runs the hipBLASLt GTest suite
  and TensileLite's GEMM tests on a gfx1250 machine, comparing results against a CPU reference. It
  is the only place in this repository's CI where kernels StinkyTofu produced actually run. It was
  added in [PR #12983](https://github.com/ROCm/rocm-libraries/pull/12983) (October 2026), and it is
  not a required check.

That last lane is valuable, and it is integration-level by nature. It checks the GEMM answers for
the shapes its suites sample, under whatever timing those runs happen to have. A hazard that fires
in one launch out of a few thousand will usually pass it.

### Performance and Benchmarking Testing

Much of StinkyTofu exists to make gfx1250 kernels faster: the DAG scheduler, the wait-count and
wait-ALU reduction rules, the prefetch passes. No lane in this repository measures the performance
of StinkyTofu's output on a pull request.

| Item | Detail |
| --- | --- |
| Stack layer | Compiler-adjacent component (kernel optimizer) |
| What exists in-tree | `EstimateAsmCyclesPass`, a cycle estimate used for remarks and tuning, not a regression signal |
| Automated measurement on a pull request | None |
| Gating approach | No performance gate |

A change that keeps every test green but costs throughput would be seen, if at all, by hipBLASLt's
own performance measurement, a day or more later and by someone other than the author; see
[hipBLASLt's performance section](../../projects/hipblaslt/TESTING.md#performance-and-benchmarking-testing).
Until something better exists, the pull request description is the only place a performance claim
can be checked, so a change to scheduling or wait reduction should carry before-and-after numbers.

### How this one was learned: a gate that turned a protection off

> [!IMPORTANT]
>
> **A small change that switched a hazard protection off for most gfx1250 kernels merged with every
> test green, because the tests checked that the code did what its author meant, and nothing
> checked whether the hardware agreed; the fix came from someone who reproduced the failure on real
> hardware.**
>
> <details>
> <summary>The full account: four pull requests, ten weeks, and what the tests could see</summary>
>
> In July 2026, [PR #9493](https://github.com/ROCm/rocm-libraries/pull/9493) extended expert
> scheduling mode 2 from a region of the kernel to the whole kernel. From then on, StinkyTofu's
> `InsertWaitAluPass` was the only thing protecting certain register hazards in every gfx1250 kernel
> that used the mode.
>
> Nine days later, [PR #10003](https://github.com/ROCm/rocm-libraries/pull/10003) added a switch,
> `EnableESM2TrackValuVsrc`, controlling whether the pass tracks the registers that vector and
> matrix instructions *read*, and turned it on only for sparse kernels. Tracking reads is what
> protects a matrix instruction's inputs from being overwritten by a later load before the matrix
> unit has read them. The change was 67 lines. Its tests looked reasonable: the existing FileCheck
> fixture for this hazard was updated to pass the new flag, and a new TensileLite unit test checked
> how the flag was derived. Both passed. But the fixture now ran with tracking on, so no test looked
> at the dense-kernel default, where tracking was off.
>
> A week later, [PR #10215](https://github.com/ROCm/rocm-libraries/pull/10215) added a temporary
> switch to suppress another of these waits. Its description said the switch defaulted to off; the
> diff set it to `true`. Nothing failed, because no test ran the pass with the switch in its shipped
> setting.
>
> On 25 August, [PR #11153](https://github.com/ROCm/rocm-libraries/pull/11153) found the second one
> the hard way. A Stream-K work-queue counter was left dirty in about 40% of runs on gfx1250
> hardware. The author built a minimal standalone reproducer, narrowed the suppression to the cases
> the scheduler's spacing guarantee actually covers, and added two regression tests: a FileCheck
> fixture at the pass level, and a rocisa test through the real production path. Then they reverted
> the fix locally and confirmed both tests failed for the stated reason. That is the pattern this
> document most wants to see copied.
>
> The first one surfaced later, as wrong GEMM results on gfx1250 in roughly one launch in two
> thousand, tracked as ROCM-31905. A rebuild with expert scheduling mode 2 turned off made the
> failures disappear, and bisection pointed to #10003. [PR #12170](https://github.com/ROCm/rocm-libraries/pull/12170)
> (2 October) re-enabled the protections, removed #10215's suppression, fixed a wrong register-field
> bit position in the hardware model, and added per-rule switches with measurements for each rule.
>
> It would be easy to read this as a review failure, and that would miss the point. Each change was
> small, plausible and tested in the way the suite makes easy. What the suite could not do was
> notice that a protection had been switched off, because its tests describe intended behavior, and
> the intent itself was what changed. Two things would have caught it. One is a behavior test for
> every protection switch in both settings, with the hazard present, asserting that the wait is
> emitted wherever the switch is off in production. The other is an independent check of the final
> kernel, either by execution or by a separate hazard checker. #11153 was the only step in this
> story that checked against hardware, and it is the one that found a bug.
>
> One question from this history is still open. #12170 also added two rules, now on by default for
> gfx1250, that let a wait carry a larger count by assuming the hardware retires certain operations
> in issue order across the matrix and vector pipelines. An independent review has questioned that
> assumption. It is settled by a small hardware test, and until then it is the reason the
> [production-configuration gap](#what-the-tests-run-versus-what-production-runs) above matters.
>
> </details>

## Pre-submit / CI Gates

This section covers GitHub Actions only. As of 19 October 2026, the required checks on `develop` are
`Component CI Summary`, `Multi-Arch CI Summary`, `pre-commit` and `Math CI Summary`; this document
does not describe the last one (see [Scope](#stinkytofu-testing-strategy)).

| Lane | Defined in | Triggered by | What it runs | Hardware | Gates? |
| --- | --- | --- | --- | --- | --- |
| `Component CI: StinkyTofu` | [`component-ci-stinkytofu.yml`](../../.github/workflows/component-ci-stinkytofu.yml) | `shared/stinkytofu/**` | Linux: GCC build, amdclang build, full CTest, pip-install test, AddressSanitizer build and CTest without the `python` label. Windows: shared and static builds with CTest, pip-install test | CPU only | **Yes**, via `Component CI Summary` |
| `Component CI: rocISA` | [`component-ci-rocisa.yml`](../../.github/workflows/component-ci-rocisa.yml) | `shared/stinkytofu/**` or rocisa | rocisa's tests against a pip install, plus shared and static plugin configurations; Windows | CPU only | **Yes**, via `Component CI Summary` |
| Multi-Arch CI | TheRock, via `therock-multi-arch-ci.yml` | Any change; the selector maps StinkyTofu to `tensilelite` | TensileLite's suite against TheRock's build | GPU runners | **Yes**, via `Multi-Arch CI Summary` |
| `clang-tidy` | [`clang-tidy.yml`](../../.github/workflows/clang-tidy.yml) | StinkyTofu C++ sources, `.def` files, `.clang-tidy` | `invoke build`, then `invoke tidy` over the whole project | CPU only | No; not in the required list |
| gfx1250-strict hardware | [`gfx1250_hipblaslt_tensilelite_test.yml`](../../.github/workflows/gfx1250_hipblaslt_tensilelite_test.yml) | Non-draft PRs from ROCm organization members touching StinkyTofu, hipBLASLt and related paths; nightly | hipBLASLt GTest suite and TensileLite GEMM tests, built for gfx1250-strict | gfx1250 | No |
| `pre-commit` | `.pre-commit-config.yaml` | Every PR | Whitespace and end-of-file fixes, `clang-format`, `black` | CPU only | **Yes** |

Two details are easy to miss. The `clang-tidy` workflow's own comments say it runs on every PR so its
check is registered, but it is not in the required list for `develop`, so a clang-tidy failure does
not block a merge today. And `component_ci.py` triggers the StinkyTofu lane on changes under
`shared/stinkytofu/**` and on a few shared CI files, but not on `component-ci-stinkytofu.yml`
itself, so a pull request that changes only the lane's definition does not run the lane.

### Validation Gates and Ownership

| Validation area | Required before merge | Owner | Responsibility |
| --- | --- | --- | --- |
| Build, Linux (GCC and amdclang) and Windows | Yes | Component team | Keep both toolchains and both platforms building with warnings as errors |
| Unit and FileCheck tests | Yes | Component team | Create, maintain, review |
| Python binding and pip-install tests | Yes | Component team | Keep bindings and packaging working |
| AddressSanitizer build and tests | Yes | Component team | Keep the sanitizer run green |
| rocisa tests against StinkyTofu | Yes | Component team with the TensileLite team | Keep the consumer build and plugin paths working |
| TensileLite suite | Yes, through Multi-Arch CI | TensileLite team | See the TensileLite document |
| clang-tidy | No | Component team | Fix findings; the lane is red when they exist, but does not block |
| gfx1250-strict hardware run | No | hipBLASLt and TensileLite teams | Read failures on StinkyTofu PRs; they may be the only execution signal |
| Adaptor emission-consistency tests | No; not run by any workflow found | Unassigned | Run locally when touching the conversion code |
| Code coverage | No; not collected in CI | Unassigned | See [Coverage](#coverage) |
| Performance | No | Component team | Put numbers in the PR description |
| Shared validation infrastructure | N/A | TheRock team | Provide shared build and test infrastructure |

### PR Test Classification

**Trusted gate.** A failure here is a real problem with the change.

- `Component CI: StinkyTofu`, on Linux, under AddressSanitizer, and on Windows
- `Component CI: rocISA`
- TensileLite's suite through Multi-Arch CI
- `pre-commit`

**Informational.** Worth reading, cannot block a merge.

- `clang-tidy`
- The gfx1250-strict hardware lane. For a change to scheduling, waits or hazards, this is the only
  execution-based signal available, so it deserves more attention than "informational" suggests.

**Unstable / flaky.** Not tracked as a category, and there is no `UNSTABLE` tag. See below.

### Flaky Test Policy

StinkyTofu's own tests are deterministic by construction: no hardware, no timing, no network. No
flaky-test tagging exists, and none has been needed so far. Two environment dependencies are handled
deliberately rather than by marking tests flaky:

- `ComgrProbeTest` skips, rather than fails, when the installed `amd_comgr` does not list the
  target instruction set.
- The Python tests share a CTest resource lock, because one of them briefly writes a probe file into
  the source tree.

If a test does turn flaky, it is a defect in the test. It should get a ticket and be fixed, not
retried until it passes.

### Known Bugs and Expected Failures

| Mechanism | What it suppresses | Ticket linkage | Detects its own fix? |
| --- | --- | --- | --- |
| `XFAIL` in a FileCheck fixture | Nothing. In `stinkytofu-check`, `XFAIL` means "the command is expected to exit non-zero" and the `CHECK` lines are still verified; the two fixtures that use it test an error diagnostic | Not applicable | Not applicable; these are error-path tests, not quarantined bugs |
| `GTEST_SKIP()` | A few cases that need an instruction a toolchain lacks, or a build-time cluster-barrier flag | None; these are environment and build-option guards, not bugs | Not applicable |
| Module-level `pytest.skip` | The intrinsic-binding tests when the binding is not built | None; environment guard | Not applicable |
| `skip-stinkytofu` marks in TensileLite configurations | Seven gfx1250 configurations, when TensileLite uses the adaptor backend | None; reason "Not yet supported in stinkytofu backend" | **No.** The configurations never run on that backend, so nothing notices when support arrives |
| [`docs/known-issues.md`](docs/known-issues.md) | Documented limitations (block-local dead-code elimination, conservative store waits) | None | Not applicable; these are design limits |

No GTest case uses a `DISABLED_` prefix. The `skip-stinkytofu` marks are the one entry here that
behaves like a quarantine list, and they would be easier to retire if each carried a ticket.

### Coverage

Code coverage (how much code the tests execute) and test coverage (how much of the intended behavior
is checked) are different things. For a component like this one there is a third distinction: a
test can confirm that a pass emits the wait its author intended, and still not show the wait is the
one the hardware needs. Coverage numbers measure only the first.

| Scope | Tool | Measured in | Enforced |
| --- | --- | --- | --- |
| StinkyTofu C++ library | `STINKYTOFU_CODE_COVERAGE=ON` and `invoke coverage` (LLVM source-based coverage, or gcov with GCC) | **Nowhere in CI**; locally only | No |

The tooling is already in good shape: `invoke coverage` runs the whole CTest suite with one profile
per process, so unit tests, FileCheck subprocesses and pytest all contribute, and writes both an
HTML report and an `lcov` file suitable for upload. No lane runs it, and
[`codecov.yml`](../../codecov.yml) has no StinkyTofu flag, so there is no number to quote today.
Wiring it up is on the [roadmap](#improvement-roadmap). A floor should wait until a few runs have
shown what the number is.

### Nightly Validation

- **gfx1250-strict hardware lane**, nightly on `develop`, with the same suites as its per-PR run.
- StinkyTofu's own suite does not run on a schedule; it runs on every pull request that touches
  it.

### Supported Configurations

Where StinkyTofu's tests run. This describes validation of this component's tests, not every
platform it builds on.

| Configuration | Validation level | Frequency | Notes |
| --- | --- | --- | --- |
| Linux, amdclang, shared library | Full CTest suite | PR | Primary lane |
| Linux, GCC | Build only | PR | Catches compiler-specific breakage; tests run on the amdclang build |
| Linux, AddressSanitizer | CTest without Python tests | PR | See [Sanitizer Coverage](#sanitizer-coverage) |
| Windows, shared and static | Full CTest suite | PR | |
| Target architecture gfx1250 | All FileCheck fixtures | PR | The only architecture StinkyTofu targets today |
| Target architecture gfx1250-strict | One FileCheck fixture; hipBLASLt and TensileLite on hardware | PR (hardware lane, not gating) and nightly | |

**Explicitly not tested:** any target architecture other than gfx1250, which is expected while it is
the only one supported, and any execution of StinkyTofu output outside the gfx1250-strict hardware
lane.

## Sanitizer Coverage

| Sanitizer | What it catches | Where it runs | Gating |
| --- | --- | --- | --- |
| **AddressSanitizer** (with leak detection) | Heap and stack overflows, use-after-free, leaks, in the library, tools and tests | Every PR, Linux, `invoke asan` | **Yes** |
| ThreadSanitizer | Data races | Nowhere; no build option | No |
| UndefinedBehaviorSanitizer | Undefined behavior | Nowhere; no build option | No |

`STINKYTOFU_ENABLE_ASAN` instruments every target before any is defined, so the library, tools and
tests are instrumented together. The Python tests are excluded from the sanitized run, because
loading an instrumented extension into an uninstrumented Python interpreter fails; the lane's comment
records why and what fixing it would take.

**Explicitly not covered:** data races (several registries, such as the named-pass and intrinsic
registries, are process-global), undefined behavior, and the Python bindings under a sanitizer.

## Static Analysis

StinkyTofu has more static analysis than most of its neighbors, and it is well configured.

- **clang-tidy.** [`.clang-tidy`](.clang-tidy) enables the bug-finding families (`bugprone-*`,
  `clang-analyzer-*`, `misc-*`, `performance-*`), leaves out style-only families on purpose, records
  a reason beside every exclusion, and sets `WarningsAsErrors: '*'`. It runs on pull requests that
  touch StinkyTofu's C++ sources or `.def` files, through `invoke tidy`. It is not a required check.
- **Compiler warnings.** `invoke build` always builds with `-Wall -Werror`, through a single
  `stinkytofu-warnings` interface target, so the CI builds fail on a new warning.
- **Formatting.** `clang-format` and `black`, through `pre-commit`, which does gate.

The remaining gap is small and procedural: making `clang-tidy` a required check, or deciding
explicitly that it should stay advisory.

## For New Contributors

When adding or modifying functionality:

1. Understand which of the [Key Quality Concerns](#key-quality-concerns) your change touches.
2. Add the appropriate level of validation (see below).
3. For a bug fix, add a regression test that fails without the fix. Revert the fix locally and
   confirm the test fails for the reason you expect.
4. Make sure the required checks pass, and read the gfx1250-strict hardware lane's result even
   though it does not gate.
5. Update this document if the testing strategy changes.

### Choosing the Right Test Type

- **Is it a pass's behavior on some input?** Add a FileCheck fixture in `tests/filecheck/`, with
  targeted `CHECK` lines on the instructions that matter. Pass the options production uses on the
  `RUN` line, or use `-O<N>` with the relevant module options.
- **Is it a data structure, an analysis, or logic inside a pass?** Add a GTest case in the matching
  `tests/unit/` subdirectory. The static test library gives you access to internal helpers.
- **Does it add, remove, weaken or gate a wait or hazard protection?** Add a behavior test with the
  hazard present, in both settings of any switch, that fails if the protection is removed. Say in
  the PR what evidence shows the change is safe on hardware. See
  [Coverage Expectations by Change Type](#coverage-expectations-by-change-type).
- **Does it change the hardware model** (instruction definitions, hazard distances, register
  fields)? Cite the source of the new value, an ISA document section or a hardware test result, in
  the commit or PR.
- **Is it in the Python bindings?** Add a pytest case in `python_module/tests/`.
- **Is it in the rocisa conversion code, or visible through rocisa?** Add a test under
  `projects/hipblaslt/tensilelite/rocisa/test/`, and run the adaptor's emission-consistency tests
  locally.
- **Is it performance-sensitive?** Measure it yourself and put the numbers in the PR.

## Coverage Expectations by Change Type

- **New pass or new pass behavior.** FileCheck fixtures for the behavior, and unit tests for any
  non-trivial internal logic.
- **Change to a wait or hazard protection.** A behavior test that fails if the protection is removed,
  run with the production option set; both settings of any switch; and written evidence that the
  change is safe on hardware. A protection's default should be on. A proposed `hazard-safety`
  change class in the hipBLASLt PR-quality skill, in
  [PR #13528](https://github.com/ROCm/rocm-libraries/pull/13528), describes this in more detail; it
  is under review and is not yet policy.
- **Change to scheduling or wait reduction for performance.** The above, if any wait moves or
  disappears, plus before-and-after measurements.
- **Hardware model change.** A test that exercises the changed entry, and a cited source.
- **Parser or serialization change.** FileCheck fixtures that round-trip the affected syntax, and a
  `FuzzTest` case if the change touches error handling.
- **Bug fix.** A regression test that fails without the fix.
- **New exported API.** A case in `api_tests`.
- **Build or packaging change.** The pip-install test and the Windows lane must pass.

## Why We Test This Way

**The design makes unit testing cheap, so it carries the load.** A pass is a function from IR to IR,
the IR has a text form, and nothing needs hardware. A test costs a few lines of input and a few
`CHECK` lines, and the whole suite runs in minutes. That is the right foundation, and it is why the
suite is large.

**Correctness here has two halves, and unit tests cover one of them.** The first half is "the pass
does what its author intended": the wait goes where the rule says, the scheduler respects the
dependency, the parser reads the syntax. Unit and FileCheck tests are excellent at that. The second
half is "the rule is what the hardware requires". That depends on a hardware model written by hand
from documentation, and in expert scheduling mode 2 the hardware does not check the hazard itself.
A test written from the same understanding as the code cannot catch a mistake in that
understanding. The [write-up above](#how-this-one-was-learned-a-gate-that-turned-a-protection-off)
is that gap, observed.

**The failures that matter are rare and timing-dependent.** A missing wait produces a wrong answer
only when the hardware happens to be slow enough to expose it, which can be one launch in thousands.
More runs of the same GEMM tests help less than they appear to. The strongest defense is by
construction: an independent check that every hazard in a final kernel has a covering wait or enough
distance, plus an execution comparison where the scheduled and unscheduled kernels must produce
identical results. Neither needs new testing machinery inside StinkyTofu's suite; both are on the
[roadmap](#improvement-roadmap).

**The component is young and moving fast.** Real development began in early 2026, and it has grown
quickly into a component every gfx1250 GEMM kernel depends on. That pace is a reason to make the
tested configuration match the shipped configuration, and to keep switches that turn protections off
under explicit test.

## Key Quality Concerns

The things that must not break, in rough order of how much damage a miss does.

**1. Hazard and wait correctness.** A missing or too-weak wait in expert scheduling mode 2 produces
silent wrong answers. Validated by FileCheck fixtures and unit tests of the wait and hazard passes,
and indirectly by the gfx1250-strict hardware lane. Not independently verified.

**2. Scheduling correctness.** The scheduler must not reorder instructions across a dependency,
including implicit ones such as the execution mask or condition codes. Validated by DAG scheduler
unit tests and fixtures, and by the implicit-dependency fixtures.

**3. Performance of the generated code.** It is the reason the scheduler exists. Not measured on a
pull request.

**4. Fidelity of parsing and emission.** Assembly text in, assembly text out, without silently
changing an instruction. Unknown instructions are kept as opaque text with a warning, so passes
cannot see their effects; this matters most for the raw-assembly path used with `stinkytofu-opt`.

**5. Memory safety and lifetime.** Validated by the per-PR AddressSanitizer run.

**6. Consumer compatibility.** rocisa and TensileLite build against StinkyTofu's exported API and
plugin interface. Validated by `api_tests`, the plugin tests and `Component CI: rocISA`.

## Release Validation

StinkyTofu is released as part of hipBLASLt and hipSPARSELt, not on its own, so its release
validation is their release validation; see
[hipBLASLt's release validation](../../projects/hipblaslt/TESTING.md#release-validation). From
StinkyTofu's side, three things are worth doing before a release:

- review the gfx1250-strict hardware lane's recent nightly history on `develop`;
- review any switch that turns a protection off in production, and the evidence each is safe;
- walk the [Known Risks and Gaps](#known-risks-and-gaps) table.

## Dependencies and Validation Handoffs

| Dependency or consumer | Owning team | How it is validated | Known gap |
| --- | --- | --- | --- |
| ROCm compiler toolchain and `amd_comgr` | Core ROCm teams | Consumed through TheRock's SDK in CI | A toolchain change surfaces as a StinkyTofu test failure |
| TheRock (build and shared test infrastructure) | TheRock team | Supplies the ROCm SDK and Multi-Arch CI | |
| rocisa (`_rocisa`, which links StinkyTofu and compiles its conversion code) | TensileLite team | `Component CI: rocISA` on every StinkyTofu change | The conversion code is tested only end to end, and the adaptor tests that cover it run in no CI lane found |
| `rocisa_stinkytofu_adaptor` | TensileLite team | Its own tests, run locally through `test.sh` | Not run by any GitHub Actions workflow found |
| TensileLite and hipBLASLt | `@ROCm/hipblaslt-reviewers` | Their suites; see [hipBLASLt](../../projects/hipblaslt/TESTING.md) and [TensileLite](../../projects/hipblaslt/tensilelite/TESTING.md) | Their GEMM tests on gfx1250-strict hardware do not gate a StinkyTofu change |
| hipSPARSELt | hipSPARSELt team | Its own suites | Not described here |

## Improvement Roadmap

Ordered by value per unit of effort. Each item is a proposal for the StinkyTofu team to accept,
reshape or reject.

The hipBLASLt testing plan proposes how kernel testing is divided across StinkyTofu, TensileLite, a
kernel corpus layer, and hipBLASLt; see *Who tests what* under
[hipBLASLt's Improvement Roadmap](../../projects/hipblaslt/TESTING.md#improvement-roadmap). Under that
split, StinkyTofu owns one question: does its transformation preserve what a kernel computes, and
protect every hazard on the hardware it targets? Testing GEMM results across shapes belongs to the
corpus layer, which tests every shipped kernel however it was generated.

The corpus layer has started, as a harness developed under AIHPBLAS-3894 (see *Kernel corpus
testing* in the same roadmap). It has not run on gfx1250 yet. Once it does, its gfx1250 kernels,
about 3,000 per variant, are the natural set for the differential execution and hazard-checker items
below. Its planned gfx1250 stress profiles repeat each kernel many times beside a double-precision
workload, the condition that exposed ROCM-31901. Those repeats are a backstop. A timing-dependent
hazard is best caught by construction, which is why the independent hazard checker below matters
more than more repeats.

### Near term, cheap and unblocking

1. **Test the production configuration.** Add FileCheck fixtures that run through `-O<N>` with the
   module options TensileLite sets for gfx1250, including `EnableESM2`, and a unit test that
   `InsertWaitAluPass` behaves correctly with `gfx1250InsertWaitAluOptions` as production uses it.
2. **A behavior test for every protection switch, in both settings.** For each option or constant
   that can remove or weaken a wait, a test with the hazard present that asserts the wait in the
   protected setting and states the safety argument in the other.
3. **Run the adaptor's tests in CI.** The emission-consistency test is the only end-to-end check of
   the conversion code. Running it in `Component CI: rocISA`, or in StinkyTofu's lane, would make it
   a gate.
4. **Collect coverage in CI.** Run `invoke coverage` in the StinkyTofu lane and upload to Codecov
   under a new flag. Report only at first.
5. **Small CI and tooling fixes.** Add `component-ci-stinkytofu.yml` to its own trigger list,
   decide whether `clang-tidy` should be required, give each `skip-stinkytofu` mark a ticket, and
   have `stinkytofu-check` fail a fixture on an unexpected exit status (and on a zero exit under
   `XFAIL`).

### Medium term, the independent checks

1. **An independent hazard checker.** A separate tool, written from the hardware hazard tables
   rather than from the scheduler's own model, that reads a final kernel and confirms every hazard
   pair has a covering wait or enough distance. It needs no hardware and catches timing-dependent
   hazards without the timing having to happen. It could run over the kernels TensileLite builds in
   CI.
2. **Differential execution.** Build a set of kernels (the kernel corpus's gfx1250 kernels) with and
   without StinkyTofu's scheduling and expert mode, run both on identical integer inputs, and require
   bitwise-identical output. The check needs no reference implementation, and it is the experiment
   that confirmed ROCM-31905, done by hand.
3. **Per-rule hardware tests.** One small kernel per hazard rule, run many times on gfx1250, to check
   the hardware model against the hardware. This is how the open question at the end of the
   write-up gets settled.
4. **Unit tests for the wait dataflow.** Direct tests for `WaitDataflow` and `InsertWaitAluPass`'s
   scoreboard, where the combinations are too many to cover well through fixtures alone.

### Longer term

1. **Mutation testing of the hazard rules.** Remove or weaken each rule in turn and require some test
   to fail. #11153 did this by hand for its own fix.
2. **Coverage-guided fuzzing** of the parsers and of random instruction sequences through the
   scheduler, checking that dependencies and required waits are preserved.
3. **Fail-closed handling of unknown instructions** on the production path: treat an instruction the
   tables do not know as reading and writing everything, or reject it.
4. **ThreadSanitizer and UndefinedBehaviorSanitizer** build options and a lane.
5. **Make the gfx1250-strict hardware lane a gate** for StinkyTofu changes once its runtime and
   stability allow it.

## Known Risks and Gaps

Stated plainly, so none of these is a surprise at release time. As in the hipBLASLt document, the
Tracking column names a work item rather than an owner, and an empty cell means the gap is
acknowledged but not yet tracked anywhere.

### Verification against the hardware

| Gap | Regression risk | Impact | Mitigation today | Tracking |
| --- | --- | --- | --- | --- |
| No independent check of the hand-written hardware model or of the waits in a final kernel | High | High | Careful review; the gfx1250-strict hardware lane catches some wrong answers, after the fact for timing-dependent ones |  |
| The one execution-based lane does not gate, and samples GEMM shapes rather than hazard patterns | Medium | High | Its results are visible on the PR |  |
| An open question about two wait rules that assume in-order retirement across pipelines | Medium | High if wrong | Settled by a small hardware test |  |

### Test configuration

| Gap | Regression risk | Impact | Mitigation today | Tracking |
| --- | --- | --- | --- | --- |
| No test runs `InsertWaitAluPass` with the production option set, or the full pipeline with `EnableESM2` | Medium | High | Each rule has its own fixture; downstream suites run production |  |
| Protection switches are not tested in both settings as a rule | Medium | High | Some are, case by case |  |
| `Gfx1250Backend`'s pass order and options are not asserted by any test | Low | Medium | Downstream suites exercise the result |  |

### Test surface

| Gap | Regression risk | Impact | Mitigation today | Tracking |
| --- | --- | --- | --- | --- |
| The rocisa conversion code has no direct test, and its end-to-end test runs in no CI lane found | Medium | Medium | `Component CI: rocISA` exercises it indirectly |  |
| Wait dataflow internals and several analyses have no direct unit test | Medium | Medium | Reached through pass-level tests |  |
| Unknown instructions are kept as opaque text, invisible to hazard analysis | Low on the production path | Medium | A warning is printed |  |
| No coverage measurement in CI | Low | Low | `invoke coverage` locally |  |

### CI and tooling

| Gap | Regression risk | Impact | Mitigation today | Tracking |
| --- | --- | --- | --- | --- |
| `clang-tidy` runs but does not gate | Low | Low | The lane is visible and red on findings |  |
| A change to only `component-ci-stinkytofu.yml` does not trigger the lane | Low | Low | Changes to the lane usually come with source changes |  |
| `stinkytofu-check` passes a fixture whose command exited with an error, if the `CHECK` lines match | Low | Medium | Most fixtures print output only after all passes finish |  |
| No ThreadSanitizer or UndefinedBehaviorSanitizer | Low | High if hit | AddressSanitizer per PR |  |
| No performance measurement on a pull request | Medium | Medium | Numbers in the PR description; hipBLASLt's own measurement later |  |
| `skip-stinkytofu` marks carry no ticket and cannot notice when support arrives | Low | Low | The reason string |  |

## Owners and Review Cadence

**Document owner:** to be agreed with `@ROCm/stinkytofu-reviewers`. This draft was written by Tony
Davis (@tony-davis) to help get the document started, and the StinkyTofu team should own it from
here.

**Review this document when:**

- a new target architecture is added;
- a new test lane is introduced, or an existing lane changes what it gates;
- a wait or hazard defect escapes to a consumer, since that is direct evidence about what this
  strategy is missing;
- before a major hipBLASLt or hipSPARSELt release, alongside the known-gap review.

At minimum, revisit the [Known Risks and Gaps](#known-risks-and-gaps) tables quarterly. The measure
of whether this document is working is whether those tables shrink.

**Counts are deliberately qualitative.** As in the hipBLASLt document, suite sizes are described as
"well over a thousand" or "over a hundred" rather than exact numbers, because they change on nearly
every pull request. Dated facts about configuration ("as of 19 October 2026") are fine as written.
