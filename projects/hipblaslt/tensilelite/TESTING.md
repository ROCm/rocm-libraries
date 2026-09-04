# TensileLite Testing Strategy

- **Owner:** T.J. Alumbaugh (@talumbau)
- **Technical Lead:** Tony Davis (@tony-davis)
- **Last Updated:** 2026-09-01

> TensileLite is the Python code generator that emits hipBLASLt's GEMM kernels and the
> solution-selection library the C++ runtime loads at dispatch time. It lives inside hipBLASLt's
> directory tree, but it is consumed by more than one component today (hipSPARSELt links it
> directly, per that project's own CI configuration), and the intent is for more consumers to follow.
> That is why its testing strategy is documented separately from hipBLASLt's C++ client and library.
>
> This document covers TensileLite's Python unit and characterization suites, the `rocisa` extension
> that backs its instruction generation, the build-time validation of the logic data it produces, and
> the CI lanes specific to all of that. For hipBLASLt's own GTest client suite, sanitizer lane, static
> analysis, and the overall two-CI-system gating picture (of which TensileLite is one large part), see
> [../TESTING.md](../TESTING.md).

## Contents

- Incident write-ups, headed *How this one was learned*:
  - [Why the net came before the tests](#how-this-one-was-learned-why-the-net-came-before-the-tests)
  - [One number and three months](#how-this-one-was-learned-one-number-and-three-months)
  - [A naming drift silently dropped a working kernel](#how-this-one-was-learned-a-naming-drift-silently-dropped-a-working-kernel)
- [Unit Testing Strategy](#unit-testing-strategy)
  - [Logic-corpus consistency regression tests](#logic-corpus-consistency-regression-tests)
- [rocisa](#rocisa)
- [Build-Time Validation of Library Logic](#build-time-validation-of-library-logic)
- [Pre-submit / CI Gates](#pre-submit--ci-gates)
  - [Where these tests actually run](#where-these-tests-actually-run)
- [Coverage](#coverage)
- [Improvement Roadmap](#improvement-roadmap)
- [Known Risks and Gaps](#known-risks-and-gaps)

## Unit Testing Strategy

**Purpose.** Validate logic that can be exercised without dispatching a kernel to a physical device:
kernel and solution selection, code generation, and the library logic data that drives them. This is
where most of hipBLASLt's hardware-independent testing lives overall — see
[../TESTING.md#unit-testing-strategy](../TESTING.md#unit-testing-strategy) for how this compares to
the C++ client's much thinner unit-test story.

The code generator is pure Python producing text, so essentially all of it is testable with no GPU.
How much of that testing is *unit* testing, as opposed to characterization scaffolding, is a separate
question that this section returns to below. It matters more than the headline coverage number does.

| Item | Detail |
| --- | --- |
| Framework | pytest, orchestrated by `tox` |
| Location | [`Tensile/Tests/unit/`](Tensile/Tests/unit/) |
| Golden snapshots | [syrupy](https://github.com/syrupy-project/syrupy), `.ambr` files in per-module `__snapshots__/` directories |
| How to run | `cd tensilelite && tox -e unit` (skips the client build), or `tox -e rocisa` for the extension only |
| Coverage | `tox -e coverage-unit` measures; `tox -e coverage-gate` enforces the floors against what it wrote |

A significant part of that suite is a **characterization suite** under
[`Tensile/Tests/unit/characterization/`](Tensile/Tests/unit/characterization/), established in
[PR #7989](https://github.com/ROCm/rocm-libraries/pull/7989) and grown considerably since. It now
spans dozens of characterized module directories, each backed by one or more `.ambr` golden files
in its own `__snapshots__/` directory, plus a separate codegen harness that characterizes generated
assembly per architecture. These
are explicitly *not* specification tests: they pin down what the code does today, including latent
bugs, so that the ongoing consolidation refactor shows any unintended behavior change as a reviewable
diff rather than a silent downstream regression. Latent bugs found while characterizing are flagged in
that directory's `DECISIONS.md` rather than silently fixed, so a golden that encodes wrong behavior is
documented as such.

**Characterization coverage is not unit coverage, and the coverage numbers do not distinguish them.**
This is the most important caveat on every percentage in this document. A characterization golden
proves that behavior did not change. It does not prove the behavior is correct, and by design some
goldens pin behavior that is known to be wrong. A unit test asserts what the code *should* do. So a
line reached only by a characterization test is protected against accidental change but unverified,
and it still owes a real test.

### How this one was learned: why the net came before the tests

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
> [`KernelWriterAssembly.py`](Tensile/KernelWriterAssembly.py) is where a GEMM kernel actually becomes
> assembly, which makes it one of the highest-consequence files in the repository. It is 20,259 lines.
> Since the monorepo reorganization in April 2025 it has taken 303 commits from 78 authors, adding
> 13,810 lines and removing 7,277. That window undercounts the file, which is years older than this
> repository. Roughly one commit in four is a fix, 74 of them, and the median gap between one fix and
> the next is four days.
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
[README.md](Tensile/Tests/unit/characterization/README.md) and is worth stating here because it is
easy to get wrong: **never run a blanket `pytest --snapshot-update`.** It rewrites every golden at
once and produces a green run that proves nothing. Update the smallest node id you intend to change,
read the resulting diff, and explain the behavior change in your PR description.

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
[`Tensile/Tests/unit/test_cpu_only_switch.py`](Tensile/Tests/unit/test_cpu_only_switch.py) and
documented, with that caveat, at
[`_codegen/GPU-MOCK.md`](Tensile/Tests/unit/characterization/_codegen/GPU-MOCK.md).

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

**Suite size.** The characterization half and the pure-unit half are roughly comparable in size, and
together they run into the thousands of tests; a small minority are GPU-guarded and skip on a
CPU-only runner. All of them run in four separate CI lanes (see
[Where these tests actually run](#where-these-tests-actually-run)).

**Coverage expectation.** The enforced whole-project floor is `fail_under = 75` in
[`pyproject.toml`](pyproject.toml), which is deliberately the only place that number lives. The
comment beside it records the intent: 80% is the target, and the floor is set below the measured
value on purpose so that ordinary run-to-run noise cannot trip an exact cutoff. Recent runs measure
78.55% on the GitHub Actions lane and 78.68% in Math CI, which is close enough agreement to trust.
Per-file floors ratchet separately.

Read that number with the caveat above firmly in mind: it is union coverage across the unit and
characterization suites, so it describes how much code is *protected*, not how much is *verified*.
There is no enforced target on the unit-only share, which is the number that actually tracks the
migration. See [Coverage](#coverage) for the full breakdown.

### Logic-corpus consistency regression tests

A third category sits outside the unit/characterization split: regression tests that scan the
*entire production logic YAML corpus* for cross-file naming and metadata invariants, rather than
exercising a fixture or pinning current behavior. Two files carry this today:
[`test_PlaceholderMerge.py`](Tensile/Tests/unit/test_PlaceholderMerge.py) (sibling `DeviceNames`
consistency, `_ID<chipid>` placeholder-suffix gating) and
[`test_GpuRevisionTarget.py`](Tensile/Tests/unit/test_GpuRevisionTarget.py) (the gfx1250 v0/v1
overlay's logic-tree shape, 4 more tests behind the same gate). In spirit this is closer to
[`TensileLogic --check-all`](#build-time-validation-of-library-logic) than to a unit test: both
validate tuning data rather than code. The difference is placement. `TensileLogic --check-all` is a
mandatory build step that cannot be skipped. These are pytest tests gated on whether the raw corpus
(`library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full`, relative to the hipBLASLt root)
happens to be present, which it is not everywhere these tests run (see
[Known Bugs and Expected Failures](../TESTING.md#known-bugs-and-expected-failures) in the hipBLASLt
doc, and [CI visibility and gating](#ci-visibility-and-gating) below).

### How this one was learned: a naming drift silently dropped a working kernel

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

## rocisa

`rocisa` is the Python/C++ extension backing TensileLite's instruction generation. It lives at
[`rocisa/`](rocisa/), inside TensileLite rather than as a separate Math CI project — there is no
dedicated CODEOWNERS team or monorepo CI "project" entry for it, so it is documented here rather than
split out further.

| Item | Detail |
| --- | --- |
| Run it | `cd tensilelite && tox -e rocisa` |
| Location | [`rocisa/test/`](rocisa/test/) |
| Framework | pytest |

**StinkyTofu coupling.** `_rocisa`'s native extension links directly against `stinkytofu::stinkytofu`
(see [`rocisa/CMakeLists.txt`](rocisa/CMakeLists.txt)), and the rocisa↔StinkyTofu conversion glue
(`AllHwMappings.cpp`, `ToStinkyTofuUtils.cpp`) is compiled straight into `_rocisa` rather than into
StinkyTofu's own library. This is a deep build-time coupling, not an incidental one, and Math CI's
own trigger config reflects it: hipBLASLt's `additionalIncludedRegions` lists `shared/stinkytofu/**`
(along with `shared/rocroller/**`, `shared/mxdatagenerator/**`, and `shared/origami/**`), so a
StinkyTofu-only change (nothing under `projects/hipblaslt/**` touched) still retriggers hipBLASLt's
own `precheckin` and `preliminary`. StinkyTofu and Origami are also the two names `preliminary`'s own
diff check looks for internally, so its tests actually run rather than no-op; rocRoller and
mxdatagenerator retrigger the pipeline the same way but are absent from that internal check, so
`preliminary` no-ops on those two unless the same PR also touches tensilelite, StinkyTofu, or Origami
(see [Dependencies and Validation Handoffs](../TESTING.md#dependencies-and-validation-handoffs) in the
hipBLASLt doc for the full comparison). StinkyTofu is still a separately owned, separately gated
component (`@ROCm/stinkytofu-reviewers`), so its own dedicated suite runs in its own CI project
independent of this; the two runs are not linked to each other.

**Component CI: rocISA**, despite the name, does not run these tests: it only tests a `pip install`
of the rocisa package. See [Where these tests actually run](#where-these-tests-actually-run).

## Build-Time Validation of Library Logic

**What it does.** `TensileLogic --check-all` validates the library logic YAML before any of it is
compiled. It checks chip IDs, matrix instructions, work-group shapes, the XCC work-group mapping, and
custom kernel declarations, one file at a time. It reads YAML only, so it needs no GPU and no compiled
kernels, and it is fast. A failure stops the build. It is the only mechanism in the component that
validates tuning data rather than code, it does not look or report like a test, and it exists because
of one specific incident.

### How this one was learned: one number and three months

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
> [Performance and Benchmarking Testing](../TESTING.md#performance-and-benchmarking-testing) in the
> hipBLASLt doc). So it shipped.
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

### Why it runs in the build

Relative to the build, the logic YAML is compiler input rather than build output:
`TensileCreateLibrary` consumes it and emits kernels from it. Validating it is front-end analysis
rather than testing, and running codegen over input already known to be invalid produces output nobody
should trust. So the check is wired in as a CMake custom command in
[`HipBLASLtCodegen.cmake`](../cmake/HipBLASLtCodegen.cmake) that runs ahead of
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

**Where this stands as of PR #11447.** As of this writing,
[PR #11447](https://github.com/ROCm/rocm-libraries/pull/11447) is open and proposes moving the
sibling-`DeviceNames`, chip-ID-arch-lock, and gfx1250v0-overlay checks out of the
`_needs_logic_dir`-gated pytest tests in
[Logic-corpus consistency regression tests](#logic-corpus-consistency-regression-tests) and into
`TensileLogic --check-all`, so they run unconditionally over the whole corpus on every build. Once
that merges, the "Fold the data-consistency checks..." roadmap item below is done, and the two
CI-visibility gap rows about Math CI skipping `Tensile/Tests/unit` on YAML-only diffs and about
`_needs_logic_dir` being unconditional in TheRock close for those three specific checks. This document
has not been updated to assume that merge yet.

### What it found once it existed

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

Its known-bug list, [`Tensile/TensileLogic/known_bugs.yaml`](Tensile/TensileLogic/known_bugs.yaml), is
the best-structured quarantine in the component (see
[Known Bugs and Expected Failures](../TESTING.md#known-bugs-and-expected-failures) in the hipBLASLt
doc). Entries are keyed on the logic file path plus the solution's `SolutionNameMin`, a
content-derived name adopted in [PR #9355](https://github.com/ROCm/rocm-libraries/pull/9355) so that
keys survive library re-tuning instead of drifting with a positional index, and each entry carries a
`ticket:` field. The checker re-validates every entry and reports the ones that no longer reproduce, so
a fixed bug is detected rather than skipped forever. All 14 current entries document the same gfx950
validation drift.

`--strict-known-bugs` turns that detection into a failure, but it defaults off and nothing passes it
today, so a stale entry only warns. Enforcing it is tracked in AIHPBLAS-4196, which proposes the
right fix: a dedicated GitHub Actions job running
[`run_tensile_logic_check.py`](../scripts/run_tensile_logic_check.py) with the flag, rather than
tightening the in-build command. That ticket also names a remaining hole, which is that an orphaned
entry whose `solution_name` resolves to nothing currently matches nothing and is silently ignored, so
strict mode is not yet a complete dead-entry detector. Extending the gate to cover derived-parameter
assignment is tracked in AIHPBLAS-3575.

## Pre-submit / CI Gates

TensileLite is exercised by both of hipBLASLt's CI systems (see
[Pre-submit / CI Gates](../TESTING.md#pre-submit--ci-gates) in the hipBLASLt doc for the two-system
overview). The Math CI job that matters most for TensileLite is **`preliminary`**, the functional gate
for TensileLite, which runs two stages on gfx12, gfx90a, gfx942, and gfx950.

As of the Aug-26 2026 `rocJenkins` reorder (`bc21df82`, AIHPBLAS-4431), the stage order is: first
`tox -e unit -- Tensile/Tests/unit`, which carries no marker filter and therefore runs the entire unit
tree, characterization included, on all four architectures. Then, *only if that stage passed*,
`tox -e py3 -- Tensile/Tests "-m common"`, the GEMM selection that genuinely needs hardware. (Before
that reorder, `common` ran first and `unit` was conditional on it; if you are reading an older mirror
of this document or a stale local copy, check which order applies before trusting either stage's
gating story.)

Three conditions narrow when any of this happens, and all three are easy to miss:

1. The job diffs the change against `develop` and skips entirely when nothing under
   `tensilelite/`, `shared/stinkytofu/`, or `shared/origami/` has changed. A pull request touching
   none of those passes it without running anything. Notably, `shared/rocroller/**` and
   `shared/mxdatagenerator/**` are *not* in this list, even though both are among the paths that
   retrigger hipBLASLt's Math CI pipeline in the first place (see
   [Dependencies and Validation Handoffs](../TESTING.md#dependencies-and-validation-handoffs) in the
   hipBLASLt doc); a rocRoller- or mxdatagenerator-only PR gets `preliminary` scheduled but the job
   finds no relevant diff and no-ops.
2. The `common` stage is conditional on the *target branch*. It runs only when the PR targets
   `develop` or one of the two `hipblaslt_common_cms_*` branches. Which tests gate your change
   depends on where you aimed it.
3. The gate itself can be dropped, on top of condition 1 above. Math CI's `statusGate` carries a rule
   that removes `preliminary` from hipBLASLt's *required* gating list whenever the same pull request
   also touches rocroller specifically (no equivalent rule exists for mxdatagenerator), so a
   rocroller-touching change loses this gate silently even if it also happened to touch tensilelite,
   stinkytofu, or origami and would otherwise have run for real.

**A short-circuit from this reorder has already had a real consequence.** Two days after the Aug-26
reorder landed, an unrelated gfx1250 unit-test failure caused the `unit` stage to fail first, so the
`common`/GEMM stage — the one that would have directly exercised a StreamK register-pool regression
(the bug behind [#11335](https://github.com/ROCm/rocm-libraries/pull/11335), fixed in
[#11471](https://github.com/ROCm/rocm-libraries/pull/11471)) — never ran at all. Under the pre-Aug-26
order, `common` ran unconditionally and would have caught it directly. See the corresponding row under
[Known Risks and Gaps](#known-risks-and-gaps) below.

**`Tests/common` (real codegen, build, and execution against a CPU reference) does not run in TheRock
CI or GitHub Actions today, for any architecture.** It is easy to read the codebase as though gfx1250
were an exception that runs `Tests/common` under GPU emulation while other architectures skip it, but
that emulation branch in TheRock's `test_tensilelite.py` is unreachable code as of this writing: per
TheRock's `amdgpu_family_matrix.py` (the live family matrix actually imported by TheRock's CI-matrix
generation; a `new_amdgpu_family_matrix.py` also exists in that tree with the same fact under a
different field name, `run_tests: False` / `runs_on: {}`, but is not yet wired into anything), the
`gfx125x` family has an empty `test-runs-on` for Linux, with a `# No hardware available for testing
yet; build-only.` comment, so the entire Test stage — not just the unit tree — is skipped for that
family. This was confirmed live on PR #11447's own checks (`Test (gfx125X-dcgpu) / Configure test
matrix` skipping). So today, `Tests/common` coverage is Math-CI-only, on real hardware
(`gfx90a`/`gfx942`/`gfx950`/`gfx12`), and does not run anywhere in the public TheRock/GHA lanes,
gfx1250 included. See the corresponding roadmap item below, which tracks wiring it in for real
hardware.

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
| `Component CI: TensileLite coverage` | [`component-ci-tensilelite-coverage.yml`](../../../.github/workflows/component-ci-tensilelite-coverage.yml) | CPU only | No. Rolls up to `Component CI Summary`, which is not required |
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
`-m common` GEMM stage and the unit tests share the node because they run in the same job.
`tensilelite-unit-codecov` needs gfx950 for its C++ half only; the Python coverage run is along for
the ride, which is most of why that job takes about two and a half hours where the CPU-only GitHub
Actions lane takes seven minutes.

**Every one of these lanes is fail-fast, and that erases signal.** In each case one stage runs first
and everything after it is conditional on it passing: the two coverage lanes through the ordering of
tox `commands`, `preliminary` through an explicit exit-code guard between its two stages (see
[Pre-submit / CI Gates](#pre-submit--ci-gates) above for which stage now runs first), TheRock's
through plain sequential ordering. One early failure means you get no information at all from the
stages behind it, which is why a red run so often says less than it appears to.

**The goldens are asserted in three of the four lanes,** because every tox environment involved
inherits syrupy from the base `[testenv]` dependency list. `preliminary` is the one that matters,
because it gates: its unit-tree stage runs with no marker filter, so a stale golden fails a required
check. The two coverage lanes assert them as well and cannot block a merge, though the CPU-only
GitHub Actions lane is fast enough that it is usually where a stale golden surfaces first.

TheRock's installed-artifact lane is the exception. It does not install syrupy, so the suite's
`conftest.py` detects the missing plugin and skips the snapshot-using tests cleanly rather than
erroring the whole run. The tests are still collected, which is why that lane's log reads as though
characterization ran. Nothing is lost, because those goldens were asserted in the source lanes before
the artifact was built, but a reader scanning that run will see skips and should know they are
deliberate.

Three adjacent lanes run none of these tests, despite their names:
`Component CI: rocISA` only tests a `pip install` of the rocisa package; Math CI's `codecov` job is
C++ hipBLASLt coverage and is one character away from `tensilelite-unit-codecov`; and `precheckin` is
hipBLASLt's own client-suite job, unrelated to any of this despite building TensileLite as a
dependency.

## Coverage

Code coverage (how much code the tests execute) and test coverage (how much of the intended
functionality and scenarios are tested) are different things, and TensileLite is a good illustration
of the difference: it could reach high line coverage while leaving whole categories of
kernel-generation behavior unexercised.

TensileLite has a third distinction on top of those two, and it is the one most likely to mislead: a
covered line may be covered by a *unit test* that asserts intended behavior or by a *characterization
golden* that merely pins current behavior, bugs included. Every enforced number here is the union of
the two and cannot tell them apart. See [Unit Testing Strategy](#unit-testing-strategy) for why that
matters; the short version is that characterization coverage is scaffolding to be repaid, not testing
that is finished.

### The characterization-versus-unit split

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

### What is measured today

| Scope | Tool | Measured in | Enforced |
| --- | --- | --- | --- |
| TensileLite Python, unit and characterization combined | `coverage.py` via `tox -e coverage-unit` | GitHub Actions, on any change under `tensilelite/**`; Math CI measures it again for codecov | Floor plus per-file ratchet, 1 pp tolerance. Enforced in a lane that is not a required check |
| TensileLite Python, unit-only share | Same lane, reported in the split summary card | GitHub Actions | **No.** Informational only, and it is the number that tracks real progress |
| TensileLite Python, mutation score | `tox -e mutation-unit` | Nowhere; run by hand | No. Report-only pilot on eight files |
| TensileLite C++ host library | `tox -e coverage-cpp` | Math CI | Reported to codecov, not enforced |

The GitHub Actions coverage lane is CPU-only and takes roughly seven minutes. It runs the
characterization suite and the pure unit suite once each under coverage, keeping the two selections
disjoint so each line can be attributed to one or both, unions the results, and renders the
non-gating split summary card. That lane is deliberately scoped and is expected to retire once the
characterization-to-unit conversion finishes, which makes the card's characterization-only count a
rough progress bar for the lane's own retirement.

### Measuring and enforcing are separate tox environments

This is worth knowing before you try to reproduce a coverage failure. `coverage-unit` runs the suites
and writes the reports, reporting with `--fail-under=0` so it never gates. `coverage-gate` reads those
artifacts and applies both floors. The GitHub Actions lane runs them as two named steps and owns the
floors; Math CI's `tensilelite-unit-codecov` runs only the first, so it cannot be failed by a gate it
does not own.

The practical benefit is local reproduction. `coverage-gate` sets `skip_install = true` and depends
only on `coverage[toml]`, so it builds no rocisa, needs no ROCm, and runs no tests. Run
`coverage-unit` once, then re-run `coverage-gate` as often as you like against the `coverage.json`
you already have. Chasing a floor failure costs seconds rather than another pass over the full test
suite.

### Targets

Two different mechanisms carry a number, and they are easy to confuse:

- The **enforced floor** is `fail_under = 75` in [`pyproject.toml`](pyproject.toml), checked on the
  combined characterization-plus-unit dataset, alongside the per-file floors in
  `coverage-baseline.json`. This is what actually fails a run.
- The **codecov target** is 80% project coverage per flag, set in the monorepo's
  [`../../../codecov.yml`](../../../codecov.yml) for `TensileLite-Unit` and `TensileLite-CPP` along
  with every other library. No patch-coverage target is configured. Codecov's report is advisory here
  because the job that uploads it is not a required check.

80% is the direction of travel for the enforced floor, and the ratchet is how it gets there.

Neither mechanism sets a target on the unit-only share, and neither one would notice if the
characterization-only count stopped falling. Both numbers can be fully satisfied by a codebase whose
Python is entirely pinned and barely verified. Setting an explicit target on the unit-only share, so
the migration has a gate and not just a dashboard, is on the [roadmap](#improvement-roadmap).

### Scope and exclusions

Coverage measurement is Python-focused, and measured on Linux only; Linux
and Windows are not tracked separately. The exclusions in
[`pyproject.toml`](pyproject.toml) cover test, build and packaging paths
rather than product modules. The kernel writers are sometimes described as uncovered exceptions, but
they are not: `KernelWriter.py`, `KernelWriterAssembly.py` and `SolutionStructs/Solution.py` all
carry active per-file floors in the seventies. The genuinely uncovered modules are elsewhere,
including `ExperimentalLibrary.py` at zero and much of `Tensile/Components/`.

**No C++ coverage target exists** for the reasons given under
[../TESTING.md#unit-testing-strategy](../TESTING.md#unit-testing-strategy) (the C++ client's own
structural blockers). Setting one before the host-side code is linkable in isolation would produce a
number nobody could act on.

## Improvement Roadmap

Ordered by value per unit of effort, not by ambition. This list only covers TensileLite, rocisa, and
the library-logic build-time validation; hipBLASLt's C++ client and library roadmap items live in
[../TESTING.md#improvement-roadmap](../TESTING.md#improvement-roadmap).

### Near term, cheap and unblocking

1. **Make the installed-artifact lane's snapshot behavior deliberate.** Either ship syrupy with the
   installed test tree so the goldens are checked there too, or state in the lane that snapshot
   coverage is intentionally left to the source lanes. Today it is a silent skip that reads like an
   accident.
2. **Make the test suite resolve its own toolchain.** Characterization tests locate `amdclang++`
   through a bare `shutil.which`, so whether they pass depends on how the surrounding lane happened
   to order `PATH`. The same tests then behave differently in different lanes for reasons that have
   nothing to do with the code under test. Resolving the toolchain inside the tox environment removes
   a recurring source of false failures.
3. **Enforce `--strict-known-bugs` in its own lane** (AIHPBLAS-4196). The detection already exists;
   what is missing is a job that fails on a stale entry. A dedicated GitHub Actions job is the right
   home for it, because the flag cannot be turned on inside the build without failing local developer
   builds. Worth extending to orphaned entries, which are silently ignored today.
4. **Run the Python linter that already exists.** `tox -e lint` is configured and invoked by nothing.
5. **Wire `Tests/common` into TheRock's `test_tensilelite.py` for real-hardware families, not just
   the currently-unreachable `gfx1250` emulation path.** Config collection is already marker-based per
   architecture, so this is mostly (a) broadening the hardcoded `gfx1250` gate and `--gpu-targets`
   string, and (b) growing the 15-minute job timeout budget, which is sized for gfx1250's ~130 configs
   alone. This closes the "`Tests/common` runs on zero TheRock/GHA architectures" gap above and gives
   redundant, faster-feedback coverage for exactly the bug class the StreamK/#11335 near-miss
   (see [Pre-submit / CI Gates](#pre-submit--ci-gates)) exposed. This is a TheRock-repository change,
   in `test_tensilelite.py` and its family-matrix configuration, not something this document can
   implement directly.

### Medium term, the structural unlock

1. **Put a number on the characterization-to-unit migration.** The split summary card already computes
   the characterization-only statement count. Track it over time and set a target on it, so the
   migration has a gate rather than a dashboard nobody is accountable for. Without this, the union
   floors are fully satisfiable by a codebase that is pinned everywhere and verified nowhere, and the
   scaffolding has no expiry date.
2. **Fold the data-consistency checks in `test_PlaceholderMerge.py` and `test_GpuRevisionTarget.py`
   into `TensileLogic --check-all`, with one asymmetry to design around first.** Both files validate
   the same class of thing that checker already owns: logic YAML data, no code, no GPU. Today they
   only run inside the pytest suite, gated on the raw corpus being on disk (`_needs_logic_dir`), which
   is permanently false in TheRock CI's installed-artifact layout and conditionally skipped in Math CI
   on YAML-only diffs (see [CI visibility and gating](#ci-visibility-and-gating)). `TensileLogic
   --check-all` runs unconditionally wherever kernels are generated, so moving a check there closes
   the "corpus is on disk" gap unconditionally, and closes the "runs on every PR" gap for whatever
   architectures that PR's builds actually target. **[PR #11447](https://github.com/ROCm/rocm-libraries/pull/11447)
   proposes exactly this and is open as of this writing; see the note under
   [Build-Time Validation of Library Logic](#build-time-validation-of-library-logic).**

   TheRock's gfx1250 lane makes the case concretely. gfx1250 has its own gap on top of
   `_needs_logic_dir`: TheRock's family matrix has no runner wired up for `gfx125X-dcgpu`, so its
   entire Test stage is skipped, independent of whether the raw corpus is present (see
   [Pre-submit / CI Gates](#pre-submit--ci-gates) above). That is the exact architecture the
   2026-08-27 break was on, and it is a second, independent reason `test_PlaceholderMerge.py` would
   never have run there even if the raw corpus had been present. `TensileLogic --check-all` runs as a
   CMake build step, not through this test runner, so it is unaffected by that gap; TheRock does build
   a gfx1250 target (`gfx125X-dcgpu` in TheRock's `amdgpu_family_matrix.py`, build-only today pending
   hardware), so moving the check there would have exercised gfx1250's logic files in TheRock CI
   specifically, not just in principle.

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

1. **Graduate mutation testing** from a report-only pilot to a maintained signal on the modules where
   it has demonstrated value. This is the companion to the migration item above: until a module has
   real unit tests, its mutation score is the only evidence that its goldens would actually catch a
   regression rather than just execute the code.
2. **Type checking on TensileLite.** No type checking exists today despite type hints being a
   documented style rule. A large dynamically typed code generator with no type verification is a
   standing risk; see the related C++ static-analysis roadmap item in
   [../TESTING.md#improvement-roadmap](../TESTING.md#improvement-roadmap).

## Known Risks and Gaps

Stated plainly, so none of these are a surprise at release time. This section covers TensileLite,
rocisa, and library-logic validation; hipBLASLt's own C++/client gaps are in
[../TESTING.md#known-risks-and-gaps](../TESTING.md#known-risks-and-gaps). On the Tracking column, see
the note there: an empty cell means the gap is real and acknowledged but not yet tracked anywhere.

### Coverage and verification

| Gap | Regression risk | Impact | Mitigation today | Tracking |
| --- | --- | --- | --- | --- |
| Enforced coverage counts characterization scaffolding as unit testing, overstating how much is verified | High | Medium | The split summary card reports the characterization-only share, but it gates nothing |  |
| Mutation testing, the only evidence a golden would catch a regression, covers eight files and runs nowhere in CI | Medium | Medium | Manual `tox -e mutation-unit`; widening PRs are in draft. Treated as report-only | AIHPBLAS-3868 |

### CI visibility and gating

| Gap | Regression risk | Impact | Mitigation today | Tracking |
| --- | --- | --- | --- | --- |
| `preliminary` is dropped from hipBLASLt's gating list when a PR also touches rocroller, and even without that rule, rocroller and mxdatagenerator are absent from `preliminary`'s own internal diff check (only tensilelite, stinkytofu, and origami are checked), so the job silently no-ops on a rocroller- or mxdatagenerator-only change regardless of the gating-list rule | Medium | High if hit | TheRock's lane still runs the unit tree, but the four-architecture GPU coverage is lost silently |  |
| The `preliminary` stage that runs the `common` GEMM suite is conditional on the target branch | Medium | Medium | Most pull requests target `develop` and do get the full gate |  |
| As of the Aug-26 2026 reorder, `unit` runs before `common` in `preliminary`, so an unrelated unit-test failure on one architecture prevents `common` from running at all that PR. This already let a StreamK register-pool bug ([#11335](https://github.com/ROCm/rocm-libraries/pull/11335), fixed in [#11471](https://github.com/ROCm/rocm-libraries/pull/11471)) merge with no `common`-stage signal, two days after the reorder landed | High | High if hit | None observed for this ordering specifically; the pre-reorder order ran `common` unconditionally | AIHPBLAS-4431 |
| `Tests/common` (real codegen, build, execution) does not run in TheRock CI or GitHub Actions for any architecture today, including gfx1250 (see [Pre-submit / CI Gates](#pre-submit--ci-gates)); coverage of that suite is Math-CI-only | Medium | High if hit | Math CI's `preliminary` runs it on real hardware, `gfx90a`/`gfx942`/`gfx950`/`gfx12` |  |
| The same TensileLite test suite runs in four lanes, three holding a GPU only one of them needs | Low | Low | Expensive in runner capacity; the redundancy does buy independent confirmation |  |
| The installed-artifact lane silently skips the snapshot tests, since syrupy is not in the installed tree | Low | Low | The goldens are enforced upstream; the skip is stated in `conftest.py` but reads like an accident |  |
| Math CI's `preliminary` job appears to skip the `Tensile/Tests/unit` suite entirely on YAML-only diffs, running only numeric/solution-correctness checks instead | High | High if hit | None observed. Confirmed by the 2026-08-27 `develop` break: a 444-file, YAML-only PR (#11274) never ran the suite containing the sibling-`DeviceNames` check, and the resulting data bug only surfaced on a later, unrelated PR that happened to touch `.py` files |  |
| The `_needs_logic_dir` xfail (see [../TESTING.md#known-bugs-and-expected-failures](../TESTING.md#known-bugs-and-expected-failures)) is unconditional in TheRock CI, so the logic-corpus consistency checks it guards never execute there | Medium | High if hit | Math CI can still catch it when its own suite actually runs, but see the row above for when it does not | |
| For gfx1250 specifically, TheRock's `amdgpu_family_matrix.py` has an empty `test-runs-on` for the `gfx125x` family (`gfx125X-dcgpu`, build-only, no runner wired up), so the whole Test stage — including both logic-corpus consistency test files under `Tensile/Tests/unit` — is skipped outright, independent of `_needs_logic_dir` | Medium | High if hit | None; this is the exact architecture the 2026-08-27 break was on | |

### Known bugs and flaky tests

| Gap | Regression risk | Impact | Mitigation today | Tracking |
| --- | --- | --- | --- | --- |
| A stale `TensileLogic` known-bugs entry only warns, because `--strict-known-bugs` defaults off | Low | Low | The checker re-validates and reports stale entries on every build | AIHPBLAS-4196 |
| The library logic gate has no check name, no test report, and cannot be run in isolation by CI | Low | Medium | It runs on every kernel-generating build including local ones, which gives it good reach | AIHPBLAS-4196 |

### Static analysis and type checking

| Gap | Regression risk | Impact | Mitigation today | Tracking |
| --- | --- | --- | --- | --- |
| Python linting is configured (`tox -e lint`) but no CI job runs it, and `ignore = E, W` narrows it to pyflakes | Medium | Low | `black` is enforced through `pre-commit`; pyflakes-class bugs are otherwise caught in review |  |
| No type checking on TensileLite, despite type hints being a documented style rule | Medium | Medium | None. A large dynamically typed code generator with no type verification |  |
