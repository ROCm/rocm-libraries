# RFC Review — Operational Concerns (Deep Pass)

You are doing a focused operational review of a single RFC. Everything in this prompt is about what happens once the code is built, shipped, and running in downstream users' environments. Design correctness, framing, and sustainability are out of scope — those belong to `/rfc-review`.

## Reviewer stance

- Read the RFC end-to-end before forming opinions. The ops implications are usually scattered across "Implementation," "Testing," and "Open questions" — not concentrated in an "Operations" section.
- Assume the author thought about the design. Your job is to find the operational assumptions they didn't write down.
- For library code (hipDNN, MIOpen, hipBLASLt, …) the operational frame is different from a service:
  - "Production" = downstream applications running the library. You do not own the runtime.
  - "Oncall" = the maintainer reading a bug report two weeks later. They need the bug report to contain enough to act on.
  - "Observability" = what the user can capture and paste into a GitHub issue. Return codes, env-var-controlled trace logs, version dumps, `--debug` modes. There is no metrics pipeline.
  - "Rollout" = a tagged release, picked up by downstream packagers on their own schedule. No feature flag service. No gradual percentage rollout. The closest thing to a kill switch is an env var that disables a code path.
  - "Capacity" usually does not apply. Library code does not add load to a fleet. Skip the sub-area with one line if so.
- A short, sharp report beats a long, hedged one. If a sub-area has nothing real, write one line and move on.
- Calibrate severity. If everything is `blocking`, nothing is.

## Producing the report

Use `assets/templates/review-report.md`. Fill sections in this order:

1. Verdict + one-sentence justification.
2. Blocking ops concerns (if any).
3. Non-blocking ops concerns and questions, grouped by sub-area below.
4. Strengths (brief).
5. Recommended deeper passes.

**Delivery**: ask the user (via `AskUserQuestion`) whether to write the report on disk next to the RFC (`<rfc-dir>/review-ops-<YYYY-MM-DD>-<reviewer>.md`), print inline only, or both — unless they already stated a preference this turn. If writing to disk, use `git config user.name` for `<reviewer>` and report the path when done.

---

## 1. Build system impact

**What to look for**
- New third-party dependencies: name, version, license, size, transitive deps. Does it pull in Boost? Protobuf? A new compiler requirement?
- Build-time cost. New dep means longer downstream consumer CI too — not just yours. Header-only template libraries can double per-TU compile time.
- CMake target shape: new targets, new install rules, new exported symbols, new `find_package` calls. Does the new target compose with the existing superbuild?
- Toolchain requirements: minimum compiler version, C++ standard bump, new HIP/ROCm version floor, new Python version, new `cmake_minimum_required`.
- Cross-platform: does the change work on the platforms the project currently supports? Linux distros (RHEL/Ubuntu/SLES), Windows (if applicable), ARM. Any "Linux only" assumption in the RFC needs to be called out.
- Binary size: matters for embedded/distribution scenarios, container images, and any consumer with size budgets. A new dep that adds 50 MB to a previously-3 MB shared library is a real cost.
- For ROCm/GPU work: AMDGPU arch list. Does this expand the build matrix (more targets compiled = longer build)? Does it introduce kernel-compile-time work (Tensile-style instantiation, MIOpen kernel DB regeneration)?
- Reproducibility: does the build use network at build time, fetch unpinned content, embed timestamps, embed paths?

**Common failure modes**
- "We'll vendor X" with no discussion of update process, security patching, or how to keep in sync.
- New CMake `find_package` with no fallback and no `REQUIRED`/optional distinction documented.
- C++ standard or compiler bump buried in an unrelated paragraph; downstream consumers find out at upgrade time.
- New header-only library included from a hot translation unit, doubling compile time across the codebase.
- New target that pulls in the whole world via transitive link, ballooning the linked binary.
- Tensile / MIOpen-DB / kernel-cache implications waved away: "the kernels will compile on first use" — first-use latency is a user-visible problem.
- Multi-arch build cost ignored: adding one solver to one arch ≠ adding it to all arches.

**Questions to ask**
- What's the build-time delta on a clean build, on the slowest currently-supported platform?
- What's the binary-size delta on the shipped artifact? On debug builds?
- Is the new dependency `REQUIRED` or optional? If optional, what's the behavior when missing?
- What's the license of the new dependency? Is it compatible with the project's distribution model?
- Does this change the minimum supported compiler / ROCm / CMake version? If so, what's the deprecation story for users currently on the older floor?
- For ROCm: does this change `AMDGPU_TARGETS`? How does the build matrix change in CI?
- Will downstream consumers (TheRock, packagers, internal users) need to change their build invocations or env? Have they been consulted?

---

## 2. Packaging & distribution

**What to look for**
- Install layout: new files under `lib/`, `include/`, `bin/`, `share/`. New CMake config files (`<project>Config.cmake`). New pkg-config files. Anything moved or renamed.
- Package metadata: CPack components, RPM spec, debian control, conda recipe, pip wheel. If the project ships any of these, the change probably touches them.
- Versioning: does this change SONAME? `LIBRARY_VERSION`? Package version semantics? A SONAME bump cascades through every downstream binary.
- Symbol export: new public symbols, new headers in the install include set, new `__attribute__((visibility))` decisions. Once exported, you own them.
- Headers leaking implementation: a new public header that `#include`s a third-party header forces every downstream consumer to also have that third-party header available.
- Optional components: is the new piece always installed, or is there a `-DHIPDNN_ENABLE_X=ON` knob? What's the default and why?
- Downstream install impact: does `apt install hipdnn` get bigger? Does `pip install` pull more wheels? Does the user need to install a new `-dev` package to use a previously-self-contained header?

**Common failure modes**
- Public header gains a `#include <new_third_party.h>`, silently making the dep transitive for every consumer.
- New install file with no CMake `install()` rule (works in dev, missing in packages).
- SONAME unchanged across an ABI break, or bumped across an ABI-compatible change. Both poison the package ecosystem.
- New tool installed in `bin/` with a generic name (`tool`, `runner`) that collides on `$PATH`.
- Plugin discovery path hard-coded to a build-time `${CMAKE_INSTALL_PREFIX}`, broken for relocatable installs.

**Questions to ask**
- What's the full diff of installed files before vs after this RFC lands?
- Does this change SONAME? If yes, what's the migration for downstream binaries linked against the old SONAME?
- Are any new public headers added? Do they include any third-party header that wasn't already transitively required?
- For optional features, what's the default in source builds vs distro packages? Who owns that decision?
- How do downstream packagers (TheRock, distro maintainers) need to update their recipes?

---

## 3. CI

**What to look for**
- New jobs: count, runtime, runner type (CPU vs GPU vs specific arch). GPU runners are expensive and often the bottleneck.
- Existing-job impact: does this slow down the default PR build? Lengthen the critical path?
- Platform/arch coverage: which combinations actually get exercised? It's tempting to add a feature gated on a config that no CI job builds.
- Cache invalidation: does the change bust ccache, sccache, Docker layer cache, CMake configure cache?
- Test parallelism: do new tests share state, files, GPUs, ports? A new test that grabs `/dev/dri/card0` exclusively breaks parallel runs.
- Flakiness risk: is the new code exercised by something time-sensitive, network-dependent, GPU-state-dependent, ordering-dependent? Library tests that touch hardware are flake magnets.
- Test isolation: does a failure in the new code mask or cascade into other tests?
- Coverage of failure paths: it's easy to test the happy path; the regression bar is whether failure cases are exercised in CI.

**Common failure modes**
- "Tests will be added" with no detail. The RFC ships, the tests never do.
- New CI job that runs on every PR but only catches one specific config — adding 30 minutes for marginal coverage.
- New job on a fresh runner image that requires a heavy install step every run because the image isn't cached.
- A test added that depends on a specific GPU model present in CI but not all developer machines.
- Multi-arch matrix expanded but only one row in the matrix actually exercises the new code.
- New flake source: timing-sensitive test, test that asserts on a thread interleaving, test that polls without a backoff.

**Questions to ask**
- What new CI jobs are added? What's the per-PR runtime cost?
- What configuration coverage exists vs is added? Are there configs that get the new code at runtime but no CI exercises them?
- How is the new code exercised in CI on failure paths, not just happy path?
- For GPU-touching tests: which arches in the existing matrix get this test? Just gfx90a? All of them?
- What's the plan if a new test turns out flaky — quarantine path, retry policy, rollback?

---

## 4. Observability

For a library, observability is *what the user can capture and put in a bug report*. There is no metrics infra, no log aggregation, no traces. The questions to ask are: when this misbehaves, how will the user know? When they file an issue, will it contain enough?

**What to look for**
- Return codes / error types: are new failure modes distinguishable from existing ones? An overloaded "internal error" is hostile to debugging.
- Error messages: do they say *what* failed and *why*, or just "failed"? Do they include relevant context (shape, dtype, device, layer name)?
- Logging: project conventions for trace/debug logging. Env vars (`MIOPEN_LOG_LEVEL`, `HIPDNN_*`). Does the new code emit at appropriate verbosity, or is it silent?
- Version surfacing: can the user dump the version of the new dep / new component for a bug report?
- Debug builds / asserts: does the new code carry meaningful asserts in debug builds, or only at production-release stripped state?
- Crash diagnostics: stack traces, signal handlers, core file behavior. New code on a hot path that segfaults with no symbol info is a worst-case bug report.
- Reproducibility: can a user generate a small reproducer? Does the design include input-capture or replay tooling?

**Common failure modes**
- New error path that returns `HIPDNN_STATUS_INTERNAL_ERROR` indistinguishably from ten other places.
- Failure detected only by output correctness — silent on the API, visible only when the user's training diverges.
- Trace logging added but at a wrong verbosity level (always-on noise, or only at a level no one runs).
- Version of new component not exposed via `*GetVersion` or env dump.
- Debug build behavior diverges from release in a way that hides bugs in CI.
- New asynchronous path (thread, stream) with no way to flush/log progress when stuck.

**Questions to ask**
- When this code fails at a downstream user, what does their bug report look like? What's the minimum it would need to contain to be actionable?
- Is the failure distinguishable in the return code / error string from existing failures?
- What env var or build mode does a user toggle to get diagnostic output from the new code? Is that documented?
- For new threading / async work: how does a user diagnose a hang? Stuck where, waiting on what?
- Is the new component's version exposed somewhere the user can capture?

---

## 5. Performance in production

**What to look for**
- Hot path: does the change add work on a frequently-called API? Look for "every call" or "per kernel launch" patterns.
- Memory footprint: per-process baseline, per-handle, per-call. New caches grow unboundedly unless someone says otherwise.
- Warm-up cost: first-call latency. Kernel compilation, JIT, lazy loading. Users notice cold start.
- Tail latency vs throughput: a change that improves throughput at the cost of p99 is a different tradeoff than the RFC may admit.
- Claims vs evidence: every "this will be fast" should have a benchmark plan. "We expect X% overhead" is fine if measurement is planned; not fine if the design depends on a number nobody measured.
- Regression surface: what existing benchmark catches a perf regression in the new path? If none, propose one.
- Allocation patterns: hot-path `new` / `std::vector` resizing / string construction are common death-by-papercut sources.
- Lock contention: new mutex on a hot path multiplies the cost of every concurrent caller, not just the one being slowed.
- Cache behavior: GPU L2, host caches, kernel-arg caches, autotune caches. New code that invalidates a hot cache costs more than it appears to.

**Common failure modes**
- "Negligible overhead" with no measurement plan.
- Performance budget claimed for the new feature but no budget claimed for the rest of the system after integration.
- Microbenchmark wins that don't survive integration (the operation gets faster but the caller does ten of them now).
- New autotune / kernel-cache path that's fast in steady state but slow on first use, and CI never measures first use.
- Memory cost stated per-handle but no statement of how many handles are realistic.
- Threading change that's faster on N=8 cores but slower on N=128 (or vice versa) with no mention of the contention model.

**Questions to ask**
- What's the expected overhead on [hot path], and how will it be measured?
- What benchmark would fail if this change regressed by 10%? If none exists, what gets added?
- What's the warm-up cost (first call, first kernel launch, first inference)? Is it the user's problem, or hidden by the design?
- What's the memory footprint per [handle / call / cache entry]? Is there an upper bound, or does it grow with workload?
- For GPU work: what's the impact on occupancy, register pressure, kernel-arg size? Has it been measured on the slowest currently-supported arch, not just the newest?

---

## 6. Deployment & rollout mechanics

For a library, "rollout" is a tagged release. There is no gradual rollout, no canary, no feature flag service. But there are still levers:

**What to look for**
- Release vehicle: which ROCm release does this target? Is the timing realistic given the implementation effort?
- Version pinning: does the change require a specific min version of a sibling component (rocBLAS, hipBLASLt, HIP runtime)? Is that pin enforced at build time, at runtime, or only documented?
- Default behavior: if the new code is opt-in (env var, runtime flag, build option), what's the default and why? Default-off de-risks rollout; default-on is the only way to actually exercise it.
- Kill switch: is there a way for a user who hits a bug to disable the new code path without rebuilding? Env vars are the typical mechanism for library code.
- Dark launch: can the new code path run alongside the old one in the same release, with the old one as fallback? This is gold for any risky change.
- Reversibility: if the next release reverts this, what breaks downstream? Any new API/symbol shipped is forever.

**Common failure modes**
- "We'll ship in ROCm X.Y" with no buffer for slippage.
- New API committed without dark-launch period — once it's in a release, removing it breaks consumers.
- No env-var kill switch on a code path with non-trivial failure risk.
- Required-version-bump on a sibling component announced inside the RFC body, not in the changelog template.
- Opt-in default that means no real user ever exercises the new path until someone manually flips it (and then it breaks immediately).

**Questions to ask**
- Which release vehicle is this targeting? What's the slip risk?
- Is the new behavior default-on or default-off? What's the rationale?
- Is there an env var to disable the new path at runtime without a rebuild? If not, why not?
- What new symbols / APIs become committed surface in this release? Can any of them be marked experimental?
- If a user hits a bug in this code in production, what's their workaround until the next release?

---

## 7. Capacity & scaling

For most library code, this sub-area is one line: "N/A — library code, no per-fleet capacity model." Don't pad.

Apply it when:
- The change introduces a server process, daemon, or persistent background thread.
- The change adds a cache or resource pool sized at runtime (kernel cache, autotune DB, memory pool).
- The change makes a previously-bounded resource unbounded (per-handle state that grows with calls).
- The change shifts work between CPU and GPU, or between host memory and device memory.

**What to look for**
- Upper bound on any new growing resource. "Bounded by available memory" is not a bound — it's the symptom.
- Sharing model: per-process? Per-handle? Per-thread? Global singleton?
- Initialization cost vs steady-state cost. Per-handle init that's 100 ms is fine for one handle, fatal for ten thousand short-lived ones.
- Resource shape changes: a previously-CPU-only path now uses GPU memory; a previously-single-threaded path now spawns workers.

**Questions to ask**
- What's the upper bound on [new cache / pool / queue]? When does it evict?
- How does this scale with the number of handles / streams / devices a user creates?
- Does this shift any work from CPU to GPU (or vice versa)? What's the new resource footprint?

---

## 8. Failure modes in production

The catch-all for "what does broken look like." For library code, broken often looks like silent wrong answers.

**What to look for**
- Silent wrong-answer paths: numeric precision changes, kernel selection that returns a different result, autotune picking a wrong-but-not-checked kernel.
- Crash modes: segfault, abort, deadlock, OOM. Where on the stack, with what symbol info?
- Partial-failure: API call that fails halfway with state left in an inconsistent shape; subsequent calls may inherit the corruption.
- Blast radius: does a failure in this code take down the whole library / handle / process, or is it contained?
- Recovery: after a failure, can the user retry, reset state, destroy and recreate the handle? Or is the only fix process restart?
- Interaction with neighboring code: does failure of the new code corrupt shared state (allocator, stream, descriptor cache)?

**Common failure modes**
- Silent numerical change passed as "improved accuracy" without bit-exact comparison against the prior version on a regression set.
- New code in an error path that itself can fail, with no story for the failure-during-error-handling case.
- Cached state that survives a failure and feeds wrong data into the next call.
- A `noexcept` boundary that swallows the actual cause and surfaces a generic error.
- Crash with no symbols / stripped binary / inlined-everything making the bug report unactionable.

**Questions to ask**
- What does broken look like? Wrong answer, crash, hang, slowdown?
- What's the blast radius — single call, single handle, process?
- How does the user recover? Retry, reset, destroy-and-recreate, process restart?
- What invariants does this code preserve under partial failure? Are they documented?

---

## Domain-specific notes (apply when relevant)

- **Shim / compatibility layers**: build-time cost paid by every downstream consumer; symbol pollution (which symbols does the shim re-export?); diagnostic story when the backend fails (does the user see the shim's error or the backend's?).
- **Plugin / extension systems**: load-time cost, discovery failures (silent vs loud), plugin version mismatch behavior, error surfacing across the plugin boundary.
- **GPU / kernel work**: per-arch build matrix cost; first-use kernel compile cost; autotune DB shipping/regeneration; precision/determinism implications; behavior under driver/runtime version mismatch.
- **CMake superbuild**: composition with existing targets, install layout consistency, export set membership, transitive dependency leakage into downstream `find_package`.
