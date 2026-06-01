# MIOpen → hipDNN Forwarding Wrapper

- Contributors: Nolan Hanna, Mitch Ousdahl

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Current System Overview](#3-current-system-overview)
4. [Proposed Design](#4-proposed-design)
5. [Key Design Decisions](#5-key-design-decisions)
6. [Risks](#6-risks)
7. [Execution Plan](#7-execution-plan)
8. [Testing Plan](#8-testing-plan)
9. [Future Considerations](#9-future-considerations)
10. [Glossary](#10-glossary)

## 1. Executive Summary

### Why

Frameworks like PyTorch, TensorFlow, ONNX Runtime, and JAX/XLA already have substantial integration with MIOpen — call sites, tuning DBs, build infrastructure, CI coverage. There is external pushback against going all-in on a hipDNN backend at this juncture: the migration work for those framework teams is non-trivial, the timing is bad for several of them, and hipDNN is still maturing. At the same time, we want MIOpen consumers to start benefiting from hipDNN's engine ecosystem (performance work, fusion graphs, new architectures) without waiting for that migration. A forwarding wrapper lets us route selected calls through hipDNN behind the existing MIOpen API, decoupling the rollout of hipDNN-backed execution from any framework-side rework.

### What

This RFC proposes a thin wrapper layer in front of MIOpen that preserves the existing MIOpen C API verbatim from the consumer's point of view, while internally choosing — on a per-entry-point basis — whether to forward the call to the original MIOpen implementation or to hipDNN. The public/private library split is **always built**; behavior is governed at runtime by an env var that defaults to pure pass-through. With the default in effect, MIOpen behaves functionally as it does today: every call lands on the original MIOpen implementation, just via one extra function-call hop measured at ≈1 ns/call (§4.3.2). Opting in (per op family, via env var) routes selected calls to hipDNN instead. The split is:

- **MIOpen Private** — the existing MIOpen implementation, with the public C entry points renamed in source to a `_impl` suffix (e.g. `miopenConvolutionForward_impl`). Functionally identical to today's library, just relocated behind a private symbol surface. Ships as `libMIOpen_private.so`.
- **MIOpen Public** — a new lightweight shared library that re-exports the original public symbol names (e.g. `miopenConvolutionForward`) and dispatches each call either to MIOpen Private or to hipDNN. Ships as `libMIOpen.so` (the filename consumers link against today, SONAME preserved).

The wrapper reaches the private library via direct link-time linkage (`libMIOpen.so` → `-lMIOpen_private`). Because the public API and SONAME do not change, this work is invisible to existing MIOpen consumers — they continue to call `miopenConvolutionForward` and link against `libMIOpen.so` exactly as today. The forwarding decision happens behind that boundary.

### Investigation status

A first-cut Phase 1 prototype has been built and exercised; raw findings are catalogued in `0001a_InvestigationReference.md` (referred to throughout this RFC as "the investigation reference"). At a high level, the prototype confirms that the public/private split builds, links, runs, and is performance-neutral; surfaces one wrapper-induced test breakage with a clear fix path; and exposes a wiring gap in the hipDNN MIOpen-provider plugin that this RFC now addresses in Phase 1 by installing a parallel `_impl`-form public header alongside the regular MIOpen headers. Specific findings are cited inline below.

### Pivot summary

The design pivoted away from an earlier draft that used a build-time CMake flag (`MIOPEN_ENABLE_HIPDNN_WRAPPER`) to gate the entire wrapper, and from a Phase 4 framing of the provider's direct private linkage. The pivoted design (1) always builds the public/private split and gates routing behavior at runtime, (2) renames the public-API symbols directly in the MIOpen source (no `-include` macro header) so consumers and providers see a single, stable source representation, (3) installs the `_impl`-form header in Phase 1 so the hipDNN MIOpen-provider plugin can be rewired against `libMIOpen_private.so` early, and (4) moves the provider's direct private linkage from Phase 4 into Phase 1. Full rationale, affected sections, concerns addressed, and new risks are documented in `RFC-pivot-summary.md`.

### Phased approach

The work is broken into four phases (full detail in §7):

1. **Pass-through wrapper + provider direct private linkage.** Establish the two-library split, dispatch plumbing, and the consumer-side `_impl` header install. Rewire hipDNN's MIOpen provider to call MIOpen Private directly (link-time, against `libMIOpen_private.so`) so the loop hazard is structurally removed before any forwarding lands. Every public entry point still forwards to MIOpen Private. Validate that overhead is negligible.
2. **Selective hipDNN forwarding.** Pick a small, low-risk set of ops; route them to hipDNN via the compile-time forwarding set when `MIOPEN_HIPDNN_FORWARDING=enabled`.
3. **Env var and logging mapping.** Translate MIOpen's debug/tuning/logging surface to hipDNN's, scoped to the variables that frameworks actually use in production.
4. **Performance baselining.** Publish end-to-end performance numbers across the wrapper-pass-through and wrapper-forwarding configurations now that the provider's direct private linkage is already in place from Phase 1.

> **Open question — wrapper ownership.** Long-term ownership of the wrapper layer (the new `MIOpen Public` artifact, the routing policy module, and the env-var translation work in Phase 3) is not yet assigned. Candidates include an integration team or the FDE function. Needs alignment with @BradPepersAMD before Phase 1 exit so that on-call rotations, bug triage, and the eventual default-flip decision (§9) have a clear DRI.

## 2. Problem Statement

hipDNN is being positioned as the next-generation graph-execution surface for AMD's deep-learning stack, but MIOpen still has a large installed base and a stable C API that frameworks (PyTorch, TensorFlow, ONNX Runtime, JAX/XLA, etc.) depend on directly. We want a path for MIOpen consumers to benefit from hipDNN's engine ecosystem — performance work, new architectures, fusion graphs — without requiring those consumers to migrate their integration code or rebuild against a new header.

This wrapper is an explicitly **temporary** measure. The longer-term goal is for frameworks to consume hipDNN features directly so that we no longer need to add new functionality to MIOpen. Eventually MIOpen itself will be deprecated and the public API exposed by this wrapper will go away — though that endpoint is expected to be a long way off, is out of scope for this RFC, and is not blocked or driven by the work proposed here. In the meantime, the wrapper gives us a way to put framework calls in front of hipDNN without forcing framework teams to do migration work first.

Concretely, we want the ability to:

- Redirect specific MIOpen API calls to hipDNN at runtime, on a per-entry-point basis.
- Roll the redirection out incrementally — one operation, or one path within an operation, at a time — and roll it back just as easily.
- Prove that the forwarding layer itself adds negligible overhead before we migrate any production traffic onto it.
- Default the wrapper to a pure pass-through so consumers who don't opt into hipDNN forwarding see today's MIOpen behavior. Existing consumers (no env-var change, no code change) get today's library semantics plus an ≈1 ns/call wrapper hop measured to be in the noise (§4.3.2).
- Let hipDNN's existing MIOpen provider bypass the wrapper to avoid the round-trip cost when hipDNN is already in the call stack — wired up in Phase 1 so the loop hazard is structurally removed before any forwarding lands.

What we are **not** trying to do in this RFC:

- Change the MIOpen public API.
- Rewrite MIOpen on top of hipDNN.
- Deprecate or remove any MIOpen entry points.

## 3. Current System Overview

MIOpen exposes a flat C API through `include/miopen/miopen.h`. Each public entry point is implemented in a corresponding `*_api.cpp` file under `src/` (e.g. `convolution_api.cpp`, `batch_norm_api.cpp`, `layernorm_api.cpp`). The API layer translates handle/descriptor objects, builds a `ProblemDescription`, calls into the solver framework, and returns a `miopenStatus_t`.

Consumers link against `libMIOpen.so` and call these symbols directly. There is no shim layer between the public symbol and the implementation; `miopenConvolutionForward` in the consumer binary resolves to the function defined in `convolution_api.cpp`.

hipDNN already ships with a MIOpen *provider plugin* (`dnn-providers/miopen-provider/`) that calls **into** MIOpen the normal way. Today, the call direction is exclusively `caller → MIOpen` and `hipDNN → MIOpen` — never `MIOpen → hipDNN`.

Today's call graph:

![Current system](reference-images/miopen-shim-current-system.png)

## 4. Proposed Design

### 4.1 Runtime feature flag

The public/private split is **always built** in every MIOpen configuration. Behavior is governed at runtime by an env var, tentatively `MIOPEN_HIPDNN_FORWARDING` (name TBD; default: `disabled`).

- **`disabled`** *(default)*: Every public entry point dispatches to MIOpen Private. The wrapper hop is ≈1 ns/call (§4.3.2), measured to be in the noise on any real op. Functionally equivalent to today's MIOpen from the consumer's perspective.
- **`enabled`**: The per-entry-point routing policy in §4.4 is consulted; selected ops are forwarded to hipDNN, others remain on Private. A companion env var (`MIOPEN_DISABLE_HIPDNN_FOR`) provides finer-grained opt-out control.

Why a runtime flag rather than the build-time CMake flag from earlier drafts:

- **One artifact to build, test, and ship.** CI does not need to maintain a wrapper-off matrix in parallel with a wrapper-on matrix. The single artifact is the one that ships, which removes the divergence risk between configurations.
- **Operator-controlled rollback.** Any consumer who hits a regression can set `MIOPEN_HIPDNN_FORWARDING=disabled` without rebuilding or replacing the library. This is the practical recovery lever during the rollout window.
- **Cleaner default flip.** When the time comes (§9), changing the default is a one-line change to the env-var parser. Consumers who have already deployed the library pick up the new default on next process start; no rebuild required.
- **Same surface area either way.** The public/private split, the rename, and the wrapper code are present in every build, so there is no "the wrapper-off path is the one that works because that's the one anyone actually ships." Every consumer is exercising the same code paths.

### 4.2 Symbol surface split (architecture)

The public API symbols are split across two libraries — a thin **Public** wrapper and the existing implementation, which becomes **Private**. After Phase 1, the wrapper is a pure pass-through to Private by default; from Phase 2 onward the wrapper may also forward to hipDNN when the runtime env var opts in. The hipDNN MIOpen-provider plugin reaches MIOpen Private directly via link-time direct private linkage from Phase 1 onward (see §4.5 and the diagram in §7 Phase 1).

- **MIOpen Private**: contains the entire current MIOpen library, with each public-API entry point **renamed in source** from `miopenFoo` to `miopenFoo_impl`. The rename is applied directly to the function definitions and to internal headers, so the existing implementation TUs compile against the `_impl` names. All other MIOpen symbols (internal helpers, solver classes, etc.) are unchanged.
- **MIOpen Public**: a new, small shared library. For each public entry point, it defines a wrapper that decides where to dispatch (Private vs. hipDNN, governed by §4.1) and forwards the arguments. The exported symbol set is identical to today's `libMIOpen.so`, so no consumer code change is required.

How the two libraries are physically packaged, named on disk, and bound together is the subject of §4.3.

### 4.3 Implementation

The wrapper ships as a thin shared library that reuses the existing MIOpen `.so` filename; the implementation library is renamed.

#### 4.3.1 Packaging and link surface

| Aspect | Value |
|---|---|
| Public wrapper artifact | `libMIOpen.so` (same filename consumers link against today) |
| Private implementation artifact | `libMIOpen_private.so` |
| Symbol rename in Private | each public entry point → `_impl` suffix (applied directly in source — see §4.6) |
| Wrapper → Private binding | direct link at build time (`-lMIOpen_private`) |
| Wrapper → hipDNN binding | direct link at build time (`-lhipdnn`) |
| Consumer impact | none — same `-lMIOpen` link line, same SONAME |
| User runtime opt-in | none for pass-through (default); env-var opt-in for hipDNN forwarding (§4.1, §4.4) |

The call path is a plain function call (one indirection), the wrapper is a normal library that can be packaged and shipped through the existing channels, and consumers don't need to know the wrapper exists. The Phase 1 provider direct private linkage (§4.5) is a straightforward link-line change in `dnn-providers/miopen-provider/CMakeLists.txt`.

Investigation status (see investigation reference §1, §2, §4 for full data):
- **Confirmed viable.** The prototype produces `libMIOpen.so.1` as the wrapper and `libMIOpen_private.so.1` as the implementation, with the wrapper holding 263 public exports and the private library holding 263 `_impl` symbols (4 non-renamed survivors, none implicated in any collision). `MIOpenDriver` links and runs correctly against the split; gtest results show 261/265 binaries clean. The prototype used a `-include miopen_private_rename.h` macro-rename mechanism, which has since been replaced in this design by a direct source rename (§4.6); the prototype's symbol-surface findings still apply because the resulting symbol set is identical.
- **Symbol-rename collisions.** No collisions observed in the prototype against the renamed surface.
- **Still to be validated.**
  - SONAME / SOVERSION inheritance and `abidiff` equivalence of the wrapper-built `libMIOpen.so` against the pre-wrapper `libMIOpen.so` baseline (a snapshot taken before this work lands, since there will no longer be a wrapper-off build to compare against on each CI run). The prototype builds reuse `.so.1`; the formal CI gates listed in §6 and §8 are not yet wired up.
  - Packaging implications for ROCm distributors: shipping a second `.so` and ensuring both end up in the right paths.
    - **Coordinator:** TBD (named owner required before Phase 1 exit). Stakeholders to contact before Phase 1 exit: **TheRock superbuild maintainers, ROCm release-engineering, distro packagers (Ubuntu, RHEL, SUSE), container-image maintainers (`rocm/dev-*` images), Conda-forge ROCm, and the framework wheel teams (PyTorch ROCm, TensorFlow ROCm)**. Sign-off must be captured in a tracking ticket. Each of these consumers has a build/packaging assumption that "MIOpen ships as one `.so`"; surfacing this late turns into a release-readiness blocker. Note: under the runtime-flag design, *every* ROCm release ships the split, so this coordination is unconditional rather than gated on a "wrapper-enabled distro" subset.
  - Interaction with the existing `MIOpen_with_plugins` CMake target. The plugin-enabled build composes additional artifacts into the MIOpen install tree and must continue to work with the wrapper present. Not covered by the current investigation; remains open.

#### 4.3.2 Findings from the prototype

A prototype split build was stood up; full data is in the investigation reference (§1–§4). Headlines:

- **Build and link.** The split configuration produces the expected `libMIOpen.so.1` wrapper + `libMIOpen_private.so.1` implementation. `MIOpenDriver` and the gtest binaries link and run; smoke tests show identical solver selection and verification numerics between the split and a non-split baseline (investigation reference §1).
- **Runtime path proof.** With per-stub `fprintf` instrumentation, a single `MIOpenDriver convfp16` invocation produced 45 wrapper-traversal lines on the split build and 0 on the baseline — every public call in the run went through the wrapper hop, no hidden bypasses (investigation reference §2).
- **Test suite.** 261 of 265 gtest binaries clean on the split build. Of the 4 case failures: 2 are wrapper-induced (`test_smoke_tuning_policy` asserts on captured `MIOPEN_LOG_FUNCTION` stderr that now contains `_impl` names) and 2 are environmental (`test_db_sync` missing a `.kdb` file). 2 timeouts (`test_lrn`, `test_soft_max`) were progressing when the per-binary 1200 s budget fired — not failures (investigation reference §3). The wrapper-induced fix is tracked as a Phase 1 follow-up (see §7 Phase 1 task list).
- **Performance.** Aggregate GPU kernel time differs by ≤2% between configurations (signed direction varies by phase — i.e., noise); wall-clock is the same or slightly faster on the split build. Direct microbenchmark of the worst-case no-work API (`miopenGetVersion`) puts the wrapper hop at **≈1 ns/call (upper bound)** in steady state; cold-load cost is **+6 major page faults** and ≤1 MB RSS (investigation reference §4). The negligible-overhead premise that justifies the whole design is empirically supported.

Taken together, the prototype clears the feasibility bar for Phase 1. Outstanding items (SONAME/SOVERSION CI gates, packaging coordination, `MIOpen_with_plugins` interaction) remain as noted in §4.3.1, and the wiring gap that blocks the provider's direct private linkage is addressed in §4.5.

#### 4.3.3 Considered alternative: `dlopen`/`dlsym` + `LD_PRELOAD`

An earlier draft proposed a dynamic-loader-based wrapper (`dlopen("libMIOpen.so")` + `dlsym("miopenFoo")`, opted in via `LD_PRELOAD`) as a fallback in case the public/private split turned out to be infeasible. Investigation found that the dynamic-loader path carries real correctness, operational, and portability costs that the direct-linkage design does not — symbol resolution ordering with `RTLD_NEXT`, static-initializer ordering, fork/exec safety of cached function pointers, thread-safety of first-call resolution, lack of link-time validation of the hipDNN ABI, opaque static-analysis (`nm`/`ldd`/`readelf -d` don't see hipDNN as a dependency), environment-dependent library discovery (`LD_LIBRARY_PATH`/RPATH/`ld.so.cache`), POSIX-only API (Windows needs a separate `LoadLibrary` shim), and a function-pointer indirection on every forwarded call that defeats inlining. The previously open edge cases — `RTLD_DEEPBIND` interactions, static-linked consumers where `LD_PRELOAD` does not interpose, `setuid` binaries where the loader strips `LD_PRELOAD` — each compound that surface area.

Because the prototype (§4.3.2) cleared the split-build path's feasibility bar, the dynamic-loader path is **not pursued**. It is recorded here so that if a future hard blocker forces a revisit, the prior analysis is available rather than re-derived from scratch.

### 4.4 Per-entry-point routing policy

Each wrapper function consults a routing decision: "should this call go to hipDNN or to MIOpen Private?" In Phase 1 the answer is always "Private". As later phases add hipDNN coverage, the policy becomes more nuanced.

Routing inputs that the policy may consider:

- The entry point itself (some are forwarded, some are not).
- Argument shape (e.g. forward only certain layouts/dtypes).
- A compile-time list (the **forwarding set**) for entry points that are known-good and forwarded when `MIOPEN_HIPDNN_FORWARDING=enabled`. This list is owned and updated by MIOpen developers as each op is qualified in Phase 2 and beyond; consumers do not opt individual ops *in* at runtime, they get whatever the compile-time list says when they enable forwarding.
- An opt-out environment variable (`MIOPEN_DISABLE_HIPDNN_FOR=...`) so that a single op family can be force-routed back to MIOpen Private without rebuilding. This is the recovery lever when a forwarded op misbehaves in production and is the documented rollback path called out in §9 (the "loud" default-flip communication promises this knob will exist). `MIOPEN_DISABLE_HIPDNN_FOR=*` disables forwarding for every op while leaving `MIOPEN_HIPDNN_FORWARDING=enabled` in effect.

The routing decision is centralized in one place (a small `routing.cpp` or equivalent) so that adding/removing a forwarded op is a one-line change.

> **Why no opt-in env var.** An earlier draft also proposed an opt-in env var (`MIOPEN_USE_HIPDNN_FOR=<ops>`) so that operators could ad-hoc-enable individual ops without rebuilding. We dropped it: the compile-time forwarding set already gives MIOpen developers the surface they need to stage ops, and combining an opt-in env var with the opt-out env var creates interaction-rule complexity ("which wins when both are set, and why") that does not pay for itself. The opt-out alone gives consumers everything they need for emergency rollback, which is the only runtime knob the rollout actually depends on.

### 4.5 hipDNN MIOpen-provider direct private linkage (Phase 1)

Today, hipDNN's MIOpen provider plugin calls into `libMIOpen.so`. Once the wrapper exists, that call would land in MIOpen Public, which from Phase 2 onward might in turn forward back into hipDNN — a loop that, even if guarded against actual recursion, adds an unnecessary hop. The provider is therefore wired to call MIOpen's private API directly in **Phase 1**, before any forwarded ops land, so the loop hazard is structurally removed rather than relied on a thread-local recursion guard during the rollout window.

The MIOpen provider is changed to link directly against `libMIOpen_private.so` and call the `_impl` symbols. This routes hipDNN-originated calls straight to the private implementation when hipDNN is the original caller, so the wrapper only sits in the path when an *external* consumer (e.g. PyTorch) called MIOpen.

**Wiring gap from the investigation, and the fix.** The provider's CMakeLists already conditionally links `MIOpen_private` when that target is exported, and the prototype confirms the link selection works as intended — but the resulting plugin **links and then fails to load** because the provider's translation units `#include <miopen/miopen.h>` and therefore reference the unrenamed names (`miopenCreate`), while `libMIOpen_private.so.1` only exports `_impl`-suffixed names (`miopenCreate_impl`). The `dlopen` step aborts at the plugin's first MIOpen relocation. Additionally, `libMIOpen.so.1` carries a hard `DT_NEEDED` on `libMIOpen_private.so.1`, so the install always loads both files transitively — direct private linkage changes *which exports bind*, not *which files load*. Full evidence in investigation reference §5.

The chosen fix (selected from the three options sketched in investigation reference §5.7) is to **install a parallel `_impl`-form public header**, `<miopen/miopen_impl.h>`, alongside the regular `<miopen/miopen.h>`. The header declares the same API surface but with each function declared under its `_impl` name. The exported `MIOpen_private` CMake target carries `INTERFACE_INCLUDE_DIRECTORIES` pointing at the install location of this header, so any consumer that links `MIOpen_private` automatically gets the `_impl` declarations on its include path. The MIOpen provider plugin opts in by including `<miopen/miopen_impl.h>` in place of `<miopen/miopen.h>` for the TUs that talk to MIOpen Private.

Why this option over the alternatives:
- Option 1 from §5.7 (install the macro-rename header + auto-apply `-DMIOPEN_BUILDING_PRIVATE -include miopen_private_rename.h` via the `MIOpen_private` target) carries the same fragility surface called out in earlier drafts: any TU that mentions a `miopenFoo` symbol *as a string* (debug-tracing macros, logging metadata, embedded format strings) gets silently rewritten, and the forced-include order is load-bearing. Acceptable inside MIOpen's own build where the surface area is auditable; not acceptable to push onto consumers.
- Option 3 (export private-side aliases so `libMIOpen_private.so.1` publishes both `miopenFoo` and `miopenFoo_impl`) re-creates the exact risk the split was trying to remove — the private library becomes reachable via the public name surface, and any future static-analysis or symbol-leak check has to special-case it.

The parallel `_impl` header keeps the public include path untouched, keeps macro rewriting out of consumer TUs, and keeps the private library exporting only `_impl` names. The cost is one extra installed header file and a small amount of conditional compilation in the MIOpen provider.

**Regression bar (Phase 1 exit):** `dlopen` of the MIOpen provider plugin built against `MIOpen_private` must succeed, and `LD_DEBUG=bindings` must show the plugin's MIOpen calls binding to `_impl` symbols in `libMIOpen_private.so.1` with **zero** bindings into `libMIOpen.so.1` for the MIOpen API surface. This is referred to throughout the rest of this RFC as the **direct private linkage regression bar**.

### 4.6 Header story

Three concerns the header story has to satisfy:

1. **The public header `<miopen/miopen.h>` is unchanged.** Consumers see the same declarations, the same names, and the same include path as today.
2. **MIOpen Private's source compiles against the `_impl` names directly.** The public-API function definitions in `src/*_api.cpp` are renamed in source from `miopenFoo` to `miopenFoo_impl`, and the internal private-side declarations they consume are likewise renamed. There is no `-include` macro header, no forced rewrite at preprocess time, no `MIOPEN_BUILDING_PRIVATE` define. The renamed names are what is actually in the source tree.
3. **Consumers that need to call the `_impl` names (the MIOpen provider plugin, §4.5) get a parallel `<miopen/miopen_impl.h>` header** that declares the API in `_impl` form. This header is installed alongside `<miopen/miopen.h>` and exposed via `MIOpen_private`'s `INTERFACE_INCLUDE_DIRECTORIES`, so any consumer that links `MIOpen_private` picks it up automatically.

**Why direct source rename rather than the prototype's macro-include mechanism.** The prototype used `-DMIOPEN_BUILDING_PRIVATE -include miopen_private_rename.h` to `#define` each `miopenFoo` to `miopenFoo_impl` across every Private-side TU. The prototype showed that the mechanism works — 263/263 entry points renamed cleanly with no observed collisions (investigation reference §1, §5) — but the macro approach carries two fragility modes that direct source rename does not:

- Any third-party header included in the same TU that mentions a `miopenFoo` symbol *as a string* (debug-tracing macros, logging facility metadata, embedded format strings) gets silently rewritten.
- The forced-include order is load-bearing; any TU that subverts it picks up unrenamed signatures and confusing link errors.

Neither surfaced in the prototype, but with the direct rename the surface area collapses to zero: there is no rewrite happening, so neither fragility mode is reachable. The cost is the one-time edit to rename the public-API function definitions in the source tree; that edit is mechanical (the public API surface is enumerated in `<miopen/miopen.h>`) and the resulting source is what every reader of MIOpen Private's source from that point onward sees, which is its own readability win.

**`<miopen/miopen_impl.h>` generation and maintenance.** The parallel `_impl` header is generated from `<miopen/miopen.h>` at build configuration time (or hand-maintained with a CI check that the two stay in sync — pick deferred to implementation), so it cannot drift from the public API. Including it from inside MIOpen's own build is forbidden (the rename is already applied in source, and re-including would only add confusion); a CI check asserts no MIOpen TU includes the `_impl` header.

**Invariant: the `_impl` header is not in MIOpen's normal public include path.** It is installed under the same prefix but is only added to a consumer's include path via the `MIOpen_private` target. Consumers that find `MIOpen` (the public target) get only `<miopen/miopen.h>`. This keeps the `_impl` names invisible to consumers who have no business calling them. The leakage check from §6 (no `_impl` symbols appear in a sample consumer that links only `libMIOpen.so`) catches violations.

## 5. Key Design Decisions

Each decision below ties back to one or more bullets in the problem statement (§2). If a decision can't be tied back, that's a signal either to drop the decision or to widen the problem statement.

### 5.1 Split into Public + Private libraries instead of a single library with two symbol sets

**Ties to:** "Roll the redirection out incrementally … and roll it back just as easily" and "Eventually let hipDNN's existing MIOpen provider bypass the wrapper."

A single shared object that exports both `miopenConvolutionForward` and `miopenConvolutionForward_impl` would also work, but a split keeps the dependency direction one-way, which is what the incremental-rollout and provider-bypass goals depend on.

| | Pros | Cons |
|---|---|---|
| **Split (chosen)** | Explicit one-way dependency: Public depends on Private; Private has no knowledge of the wrapper. hipDNN's provider can link Private directly (Phase 4). Wrapper symbol footprint stays small and inspectable. Wrapper can be stripped/optimized independently of MIOpen proper. | Two artifacts to package and ship. Slightly more build-system complexity. |
| **Single `.so` with both symbol sets** | One artifact. No SONAME juggling. | The `_impl` symbols leak into anything that links against MIOpen — the provider can't cleanly bypass the wrapper without seeing both surfaces. Harder to reason about which symbol set is "the API." |

### 5.2 Runtime flag instead of a build-time flag

**Ties to:** "Default the wrapper to a pure pass-through so consumers who don't opt into hipDNN forwarding see today's MIOpen behavior" and "Roll the redirection out incrementally … and roll it back just as easily."

An earlier draft used a build-time CMake flag (`MIOPEN_ENABLE_HIPDNN_WRAPPER`, default OFF) to gate the entire mechanism. The flag-off build was bit-identical to today's MIOpen. The pivot replaces that with a runtime env var that defaults to `disabled`; the public/private split, the rename, and the wrapper code are always built. The justification for the change is that the build-time approach had three downstream problems that compounded over the rollout window:

- **CI doubled.** Maintaining a wrapper-off and a wrapper-on matrix in parallel was expected to add ~4–6.5 hours per full-arch pipeline run, based on the most recent successful MIOpen CI build (~6 hours) and TheRock Release portable Linux package builds (~4–6.5 hours). Parallelizing the matrix kept wall-clock flat but doubled compute.
- **Configuration divergence risk.** Two artifacts means two code paths that can diverge. The wrapper-off artifact is the one anyone runs in production until the default flips, so it becomes the artifact that "is supposed to work," and the wrapper-on artifact gets exercised less. The split incentive structure is exactly the wrong one for catching regressions in the path that is supposed to eventually become the default.
- **Default flip becomes a packaging migration.** Flipping the default under the build-time scheme requires every downstream distro/wheel/container to rebuild and re-publish before consumers see the new behavior. Under the runtime scheme it is a one-line change to the env-var parser; consumers pick up the new default on the next process start with no rebuild.

| | Pros | Cons |
|---|---|---|
| **Runtime flag (chosen)** | Single artifact: one build to test, one binary to ship. Operator-controlled rollback (`MIOPEN_HIPDNN_FORWARDING=disabled`) without rebuilding. Clean default flip when the time comes. Every consumer exercises the same code paths, so the pass-through path gets continuous coverage rather than being the one that "doesn't ship." | Wrapper is always present in the binary and in the call path; consumers cannot get a literally byte-identical-to-today's-MIOpen artifact. Mitigation: the ≈1 ns/call wrapper hop is in the noise on any real op (§4.3.2); `abidiff` against a pre-wrapper baseline asserts the public symbol set is identical. |
| **Build-time flag** | Flag-off artifact is bit-identical to today's MIOpen; clean A/B comparison for perf regression detection. | Doubled CI, configuration divergence risk, packaging migration on default flip. The bit-equivalence guarantee was load-bearing in the prior draft but is replaced here by the wrapper-hop overhead measurement plus the public ABI check. |

### 5.3 Direct linkage rather than a `dlopen`/`dlsym` + `LD_PRELOAD` wrapper

**Ties to:** "Prove that the forwarding layer itself adds negligible overhead" and "Let hipDNN's existing MIOpen provider bypass the wrapper."

The chosen design ships `libMIOpen.so` as a wrapper that link-time depends on `libMIOpen_private.so` and `libhipdnn.so`. The dispatch is a plain function call. An earlier draft kept a `dlopen`/`dlsym` + `LD_PRELOAD` design as a fallback in case the direct-linkage path turned out to be infeasible; the prototype (§4.3.2) cleared the feasibility bar, so the dynamic-loader fallback is dropped from the live design and documented in §4.3.3 as a considered alternative.

Why direct linkage:
- **No user-side opt-in.** No `LD_PRELOAD` to set, no per-process initialization to remember.
- **Plain function-call dispatch.** ≈1 ns/call upper bound (§4.3.2); easiest to argue is overhead-free.
- **Provider direct private linkage is a link-line change** (§4.5).
- **SONAME is preserved**, so consumers see no change.
- **Link-time ABI validation.** hipDNN ABI breaks surface as build failures in CI rather than `dlsym` returning `NULL` deep inside a production workload.
- **Works the same on Windows and Linux** (modulo SONAME / DLL naming) — no `LoadLibrary` shim.

The downsides of direct linkage are limited to the build-system mechanics for producing a split (renaming the implementation library, packaging two `.so` files), which the prototype confirmed are tractable. See §4.3.3 for the full catalog of why the dynamic-loader path was not pursued.

### 5.4 Phase env vars and logging separately

**Ties to:** "Roll the redirection out incrementally — one operation, or one path within an operation, at a time."

MIOpen has a substantial set of debug / tuning / logging environment variables and a logging subsystem with its own conventions. hipDNN has a different set. Mapping these requires care — some MIOpen knobs have no hipDNN analog and vice versa, some affect the call path before any forwarding decision can be made, and some affect kernel selection inside MIOpen Private in ways the wrapper can't influence. Gating the forwarding work on resolving the mapping would block incremental rollout. Phase 2 explicitly defers it (by only forwarding ops insensitive to env vars), Phase 3 takes it on as the focus.

| | Pros | Cons |
|---|---|---|
| **Phase separately (chosen)** | Forwarding rollout starts immediately on insensitive ops. Env-var mapping gets the design attention it deserves rather than being rushed. | A subset of ops is off-limits for forwarding until Phase 3 lands. |
| **Block on env-var/logging mapping first** | Single, complete forwarding story at launch. | Pushes any user-visible benefit out by months for a problem that doesn't affect early targets. |

### 5.5 Direct private linkage for the MIOpen provider in Phase 1 rather than Phase 4

**Ties to:** "Let hipDNN's existing MIOpen provider call the private MIOpen API directly to avoid the round-trip cost when hipDNN is already in the call stack — wired up in Phase 1 so the loop hazard is structurally removed before any forwarding lands."

An earlier draft deferred the direct private linkage to Phase 4. The pivot moves it to Phase 1 because (a) Phase 2 is the first phase that can actually create the loop (hipDNN provider → wrapper → hipDNN), and structurally removing the loop before that point eliminates a class of recursion-guard bugs and reasoning load entirely, and (b) the consumer-side `_impl` header that the direct linkage needs (§4.5) is also the mechanism the provider plugin needs to load successfully against `libMIOpen_private.so`, so the two changes are coupled and don't benefit from being split across phases.

| | Pros | Cons |
|---|---|---|
| **Phase 1 (chosen)** | Loop hazard removed structurally before any forwarding lands; no thread-local recursion-guard needed during Phase 2. Couples the `_impl` header install (which the plugin needs to load) with the link-line change (which the plugin needs to reach the private API directly) — both land together. Provider tests exercise the direct-linked path from day one. | Couples the wrapper rollout to a hipDNN-side build change; Phase 1's blast radius is larger. Mitigation: the change is contained to `dnn-providers/miopen-provider/CMakeLists.txt` and one header include swap. |
| **Phase 4** | Phase 1 changes stay minimal. The wrapper is validated end-to-end before hipDNN's build is touched. | Until Phase 4, hipDNN-originated calls go through the wrapper unnecessarily, and once Phase 2 lands, the loop is gated by a recursion guard rather than by structure. |

## 6. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Per-call overhead from the wrapper is non-negligible for short-running ops | Low | High | **Largely mitigated by the prototype**: microbenchmark of the worst-case no-work API (`miopenGetVersion`) shows ≈1 ns/call upper-bound steady-state cost and +6 major page faults at cold load (investigation reference §4). For any real op that touches the GPU, wrapper hop is unmeasurable. This overhead is now paid in every configuration (pass-through is no longer a separate build); the measurement therefore covers every shipped consumer rather than only opted-in builds. |
| Renaming public symbols to `_impl` collides with existing internal symbols somewhere in MIOpen | Low | Medium | The renaming is mechanical and applied directly in source (§4.6); collisions surface at compile or link time. We grep the source for `_impl` suffixes on the public-API names before merging. Prototype run produced 263/263 renamed entry points with 4 unrelated non-renamed survivors and no observed collisions (investigation reference §5.4). |
| Behavioral divergence between MIOpen and hipDNN for a forwarded op (precision, edge cases, error codes) | Medium | High | Phase 2 forwards a small, well-tested set first under `MIOPEN_HIPDNN_FORWARDING=enabled` opt-in. The existing MIOpen test suite is the acceptance gate — anything that regresses against pass-through behavior is reverted. Prototype runs at the pass-through level confirm 261/265 gtest binaries clean and identical solver selection in smoke tests (investigation reference §1, §3), so the gate is in working order before any forwarding lands. |
| Environment variables silently change meaning when forwarding is on | High if unaddressed | Medium | The default of `disabled` means the env-var surface remains MIOpen-native until a consumer opts in. Phase 2 explicitly does *not* forward any op whose behavior is sensitive to env vars we haven't mapped. Phase 3 builds the mapping. |
| hipDNN ABI churn breaks the wrapper | Medium | Medium | Direct linkage means breakage surfaces at link time and is caught in CI. hipDNN ABI changes need to be coordinated with the wrapper's release cadence. |
| Consumers that statically link MIOpen lose the wrapper indirection | Low–Medium (assumption — needs confirmation) | Low | Currently flagged as low-likelihood, but this is an assumption that needs confirmation: **before Phase 1 exit, reach out to the framework and downstream-consumer teams (PyTorch ROCm, TensorFlow ROCm, internal AMD consumers) to inventory whether anyone actually static-links MIOpen today.** If any production consumer does, the risk likelihood and the mitigation both need to be re-scoped. Until then, document that the wrapper requires the shared-library build; if a static-link consumer surfaces and matters, a separate mechanism (likely link-time symbol substitution) is needed. |
| Loop / recursion if the MIOpen provider in hipDNN re-enters the wrapper | Low | High | The Phase 1 provider direct private linkage (§4.5) structurally prevents the loop: the provider links `libMIOpen_private.so` directly and calls `_impl` symbols, never touching `libMIOpen.so`. Until Phase 2 lands, the wrapper has nothing to forward to hipDNN anyway. The Phase 1 regression bar (`LD_DEBUG=bindings` shows zero plugin → `libMIOpen.so.1` bindings on the MIOpen API surface) is the CI gate. |
| Operator sets `MIOPEN_HIPDNN_FORWARDING=enabled` on a workload Phase 2 has not validated | Medium | Medium | The opt-in is intentional and per-process; the failure mode (forwarded op misbehaves) is recoverable via `MIOPEN_HIPDNN_FORWARDING=disabled` or per-op `MIOPEN_DISABLE_HIPDNN_FOR=<op>` (§4.4). Documentation lists the validated op set per release; routing tracing (Phase 2 task) makes "what got forwarded" visible at runtime. |
| `_impl` symbols or the `_impl` header leak into a consumer that links only `libMIOpen.so` | Low | High | The rename is applied directly in source; there is no rewrite macro to leak. The `_impl` declarations live in `<miopen/miopen_impl.h>`, which is reachable only via `MIOpen_private`'s `INTERFACE_INCLUDE_DIRECTORIES`. CI checks: (a) `grep -r '_impl' <staged public headers under MIOpen>` returns no matches (the `_impl` header is staged separately under the private install path); (b) sample-consumer build links only `libMIOpen.so`, includes only `<miopen/miopen.h>`, and `nm` of the resulting binary shows no `*_impl` symbols; (c) `abidiff` compares the wrapper-built `libMIOpen.so` against the pre-wrapper `libMIOpen.so` snapshot (captured before this work lands) and asserts the exported public symbol set is identical. |
| hipDNN is missing, broken, or fails at runtime | Medium | High | Behavior is defined per failure class (load-time vs per-call) with a thread-local "last forwarded error" mechanism so on-call can tell which side failed. See §6.1 immediately below. |

### 6.1 Failure-mode behavior for hipDNN forwarding

Three failure classes, each with explicit defined behavior:

- **Load-time failure** — the wrapper's hipDNN dependency cannot be resolved when `libMIOpen.so` is loaded (e.g. `libhipdnn.so` missing on the system path, or a SONAME mismatch). The loader will normally fail to load `libMIOpen.so` outright, which surfaces as a clear failure at process start rather than silently. As a defensive measure, the wrapper logs once to stderr at init if it detects partial hipDNN availability, sets an internal `hipdnn_unavailable` flag, and routes every call to MIOpen Private for the lifetime of the process (matching the `MIOPEN_HIPDNN_FORWARDING=disabled` semantics).
- **Per-call hipDNN failure** — a forwarding attempt returns a hipDNN-side error. **The failure is propagated to the caller as a translated `miopenStatus_t`. We do not silently fall back to MIOpen Private mid-execution**, because silent fallback would mask correctness regressions (hipDNN returned a wrong-but-not-error result, MIOpen would have returned right) and hide perf regressions. Operators retain `MIOPEN_DISABLE_HIPDNN_FOR=<op>` (§4.4) as the kill switch for a flaking op.
- **Error-code distinguishability.** A translated `miopenStatus_t` alone is ambiguous — on-call cannot tell from `MIOPEN_STATUS_INTERNAL_ERROR` whether to read MIOpen source or hipDNN source. The wrapper provides:
  - a thread-local accessor `miopenGetLastForwardedError()` returning the hipDNN-side status (if any) for the most recent forwarded call on the current thread, and
  - a `[hipDNN-forwarded]` prefix on the `miopenGetErrorString` message for any forwarded-error code, so log scraping is unambiguous.
  - A new dedicated error code (`MIOPEN_STATUS_FORWARDED_ERROR`) was considered but rejected for Phase 2 because it would change the visible status set for any consumer that opts into forwarding; the thread-local accessor is additive and does not. To be re-evaluated if forwarded-error patterns turn out to warrant their own bucket.

## 7. Execution Plan

The work is broken into four phases. Each phase ends with the existing MIOpen test suite green in the default (`MIOPEN_HIPDNN_FORWARDING=disabled`) configuration; phases that forward to hipDNN additionally run the suite with the relevant forwarding env vars set.

> **Estimates.** Rough per-phase effort estimates, in person-sprints:
> - **Phase 1:** 2
> - **Phase 2:** 2–4
> - **Phase 3:** 2–4
> - **Phase 4:** 1
>
> Phase 1 is the best understood — the prototype already covers most of it (investigation reference §1–§4) and the remaining work (direct source rename, `_impl` header install, provider rewire, env-var plumbing, CI gates, wrapper-induced test fix) is well-scoped. **Phase 2 onward should still be treated as quite rough**: the actual cost depends on which op families end up in the forwarding set, on hipDNN's coverage at the time, on what the framework-env-var audit (§7 Phase 3 task 1) surfaces, and on how much of the existing MIOpen env-var surface turns out to be production-relevant. Re-scope these numbers as each phase enters planning.

### Phase 1 — Pass-through wrapper, source rename, and provider direct private linkage

A prototype covering the build/link mechanics, runtime path, wrapper stubs, and performance characterization has been built (investigation reference §1–§4). The Phase 1 increment over the prototype is: convert the macro-based rename to a direct source rename (task 3), install the parallel `_impl` header (task 4), rewire the MIOpen provider for direct private linkage (task 5), add the runtime env var (task 7), stand up the CI gates (task 8), the PyTorch coverage (task 9), and the wrapper-induced test fix (task 11).

Goal: establish the two-library split, the dispatch plumbing, the runtime env-var control surface, and the provider's direct private linkage, with **all** public entry points dispatching to MIOpen Private when `MIOPEN_HIPDNN_FORWARDING=disabled` (the default). From the consumer's perspective with the default in effect, behavior is functionally identical to today's MIOpen.

![Architecture after Phase 1 (pass-through, provider direct-linked to MIOpen Private)](reference-images/miopen-shim-phase-1.png)

Tasks:
1. Add the `MIOpen_private` CMake target and wire its `INTERFACE_INCLUDE_DIRECTORIES` to expose `<miopen/miopen_impl.h>` (see task 4). The build always produces both `libMIOpen.so` (wrapper) and `libMIOpen_private.so` (implementation). CMake should emit a clear status message at configure time announcing the build state ("MIOpen wrapper: built — runtime default is pass-through; hipDNN forwarding controlled by MIOPEN_HIPDNN_FORWARDING env var") so that downstream packagers cannot miss the structural change.
2. Confirm build-system mechanics, SONAME and SOVERSION inheritance, and `MIOpen_with_plugins` interaction. *(Status: build-system mechanics and runtime path confirmed by the prototype — investigation reference §1, §4. SONAME/SOVERSION CI gate is task 8; `MIOpen_with_plugins` interaction remains open per §4.3.1.)*
3. **Direct source rename.** Rename the public-API function definitions in `src/*_api.cpp` (and the corresponding internal headers under `src/include/miopen/`) from `miopenFoo` to `miopenFoo_impl`. Mechanical edit driven by the public API surface enumerated in `<miopen/miopen.h>`. *(Status: prototype used a macro-based `-include` mechanism (§4.6 historical), produced 263/263 renamed entry points. Direct source rename replaces that and is new Phase 1 work.)*
4. **Install `<miopen/miopen_impl.h>`.** Generate or hand-maintain the parallel `_impl`-form public header alongside `<miopen/miopen.h>` (§4.6). Install it under the private include subtree and add it to `MIOpen_private`'s `INTERFACE_INCLUDE_DIRECTORIES`. CI check: a TU built against `MIOpen_private` and including `<miopen/miopen_impl.h>` resolves the `_impl` declarations; a TU built against `MIOpen` only and including `<miopen/miopen.h>` resolves only the public names.
5. **Rewire the MIOpen provider plugin for direct private linkage (Phase 1).** In `dnn-providers/miopen-provider/`, swap the link target to `MIOpen_private` and the include from `<miopen/miopen.h>` to `<miopen/miopen_impl.h>` for the TUs that talk to MIOpen. Verify `dlopen` of the resulting plugin succeeds and that `LD_DEBUG=bindings` shows the plugin's MIOpen API calls binding to `_impl` symbols in `libMIOpen_private.so.1` with zero bindings into `libMIOpen.so.1` (the §4.5 regression bar).
6. Generate the wrapper source file. Each entry point has a stub that forwards the call to the corresponding `_impl` symbol via a plain function call. *(Status: 263 stubs in place; runtime-traversal proof in investigation reference §2.)*
7. **Add the `MIOPEN_HIPDNN_FORWARDING` env var** (§4.1) and the wrapper's runtime dispatcher. Phase 1 only needs the `disabled` path to be load-bearing; `enabled` is parsed and accepted but is functionally equivalent to `disabled` because no op is in the forwarding set yet. A configuration banner is emitted to stderr once at first wrapper call summarizing the resolved mode.
8. Add the **header-leakage and ABI CI checks** described in §6 and §4.6: (a) `grep -r '_impl' <staged public headers under MIOpen install root>` returns no matches; (b) sample-consumer build links only `libMIOpen.so`, includes only `<miopen/miopen.h>`, and `nm` of the resulting binary shows no `_impl` symbols; (c) `abidiff` compares the newly built `libMIOpen.so` against a pre-wrapper `libMIOpen.so` snapshot (captured before this work lands) and asserts the exported public symbol set is identical. *(Status: not yet wired into CI.)*
9. **Stand up the PyTorch-on-ROCm CI coverage** described in §8.1 (the convnet/batchnorm/RNN test modules and the TorchBench subset) against the wrapper build with the default env-var setting. This is the highest-coverage realistic-workload signal for the pass-through validation and is reused throughout Phases 2–4. Requires coordination with the PyTorch-on-ROCm team to identify the right test subsets and to get the wrapper-built MIOpen exercised in their regression pipelines; this coordination starts now so it is in place before Phase 2 forwards anything. *(Status: not yet started.)*
10. Run the MIOpen test suite **and** the MIOpen provider tests (in `dnn-providers/miopen-provider/`) with the default env var setting (`disabled`); all must be green. *(Status: gtest pass on the prototype — 261/265 binaries clean, with 2 wrapper-induced case failures in `test_smoke_tuning_policy` (see task 11), 2 environmental case failures in `test_db_sync` (missing `gfx942.kdb`), and 2 timeouts in `test_lrn`/`test_soft_max` that were still progressing at the 1200 s per-binary budget — investigation reference §3.)*
11. **Wrapper-induced test fix** (known-issue follow-up surfaced by task 10). `test_smoke_tuning_policy` asserts on captured stderr from `MIOPEN_LOG_FUNCTION`, which now contains `_impl`-suffixed function names rather than the public names the test was written against. **Fix: relax the test assertion to accept the `_impl` form.** The alternative of normalizing the wrapper's log output back to the public name was considered but rejected: under direct source rename, the function names are genuinely `_impl` in source, so normalizing the log would be hiding the actual control flow from the test rather than reflecting it. Investigation reference §3 has the failure details.

Exit criteria (Phase 1):

1. **Functional parity (default mode).** Wrapper build passes the same MIOpen test suite + MIOpen provider tests + sample-consumer build smoke test with `MIOPEN_HIPDNN_FORWARDING=disabled` (the default) as the pre-wrapper baseline. Phase 1 explicitly does not exercise the env-var or logging mapping — the forwarded op set is empty, so env-var-sensitive paths are out of scope for this phase. Tasks deliberately select ops whose pass-through behavior is insensitive to env-var state. Env-var coverage is the focus of Phase 3. *(Status: provisionally met on the prototype for the gtest suite — 261/265 binaries clean — pending the test fix in task 11 and the environmental `test_db_sync` setup gap. Investigation reference §3.)*
2. **Steady-state wall-clock overhead.** Wrapper adds < 1% wall-clock overhead on a representative end-to-end workload, measured against the pre-wrapper baseline snapshot. This is the gate for the wrapper itself being acceptable as a permanent piece of the call path that every consumer now pays. *(Status: met on the prototype. End-to-end conv workload shows wall-clock the same or slightly faster vs baseline; aggregate GPU kernel time deltas ≤ 2% with signed direction varying by phase. Investigation reference §4.)*
3. **Per-call overhead breakdown (gate, not a target).** Measure overhead from each of: parameter translation, hipDNN graph building (where applicable in later phases — measured here so the Phase 2 number can be compared cleanly), dispatch indirection. If any single component dominates the < 1% budget, escalate before Phase 2 starts. Mitigation strategies that are pre-approved if translation overhead is a problem: caching translated graphs, caching descriptor conversions, lazy materialization of hipDNN handles. *(Status: dispatch indirection measured at ≈1 ns/call upper bound. Parameter translation and hipDNN graph building are Phase 2 measurements — there is no forwarded op to translate against in Phase 1.)*
4. **Cold-start vs warm-start compilation time.** Measure first-call latency separately from steady-state. The risk we are gating against is that hipRTC compilation (or any other one-time JIT work on either side) happens *twice* — once on a MIOpen Private path and once on a hipDNN path — for a workload that, before this work, would have compiled it only once. The exit criterion is that cold-start latency for any forwarded op family with `MIOPEN_HIPDNN_FORWARDING=enabled` is ≤ 1.10× cold-start with `=disabled`, and warm-start is within the < 1% steady-state budget above. *(Status: pass-through cold-load characterized — +6 major page faults, ≤1 MB RSS, no detectable wall-clock change. The double-compilation gate stays open until Phase 2 introduces a real hipDNN forwarding path. Investigation reference §4.)*
5. **PyTorch-on-ROCm CI signal.** The PyTorch coverage stood up in task 9 is green against the wrapper build (pass-through default) for the agreed test subset. *(Status: not yet stood up.)*
6. **Header-leakage and ABI CI gates green.** The three checks added in task 8 are wired into CI and pass. *(Status: not yet wired up.)*
7. **Provider direct private linkage regression bar (§4.5).** `dlopen` of the rewired MIOpen provider plugin succeeds, and `LD_DEBUG=bindings` shows zero plugin → `libMIOpen.so.1` bindings on the MIOpen API surface. *(Status: not yet wired up — depends on tasks 4 and 5.)*

### Phase 2 — hipDNN forwarding for selected entry points

Goal: actually forward a small, low-risk set of API calls to hipDNN. Defer environment-variable and logging concerns.

Tasks:
1. Pick the initial **compile-time forwarding set** (§4.4). Candidates are entry points where hipDNN already has full coverage and the behavior is well-understood (likely starting with a single op family). This list lives in the routing policy module and is what determines which ops are forwarded when `MIOPEN_HIPDNN_FORWARDING=enabled`.
2. Implement the routing policy module — a single source file that the wrapper consults to decide Private vs. hipDNN per call (consulting the compile-time forwarding set and `MIOPEN_DISABLE_HIPDNN_FOR`).
3. Implement the hipDNN call paths: argument translation from MIOpen descriptors to hipDNN graph + variant pack, hipDNN execution, result translation back to `miopenStatus_t`.
4. Wire the opt-out env var (`MIOPEN_DISABLE_HIPDNN_FOR=...`, §4.4) into the routing policy. `MIOPEN_DISABLE_HIPDNN_FOR=*` forces every op back to MIOpen Private while leaving `MIOPEN_HIPDNN_FORWARDING=enabled` in effect (this is the "wrapper enabled but every op back on MIOpen" rollback path documented in §9).
5. **Add routing tracing.** When `MIOPEN_LOG_LEVEL` (or a wrapper-specific equivalent) is set, every wrapper invocation logs a single line containing the entry-point name, the resolved `MIOPEN_HIPDNN_FORWARDING` mode, the routing decision (Private vs. hipDNN), and the reason (in the compile-time forwarding set, env-var opt-out, policy fall-through). This is the on-call instrument for "why did this call go where I didn't expect" investigations and is a prerequisite for confidently flipping the `MIOPEN_HIPDNN_FORWARDING` default in any environment.
6. Implement the failure-mode behavior defined in §6.1: load-time fallback to Private with a one-time warning, per-call hipDNN errors propagated (not silently absorbed), and the `miopenGetLastForwardedError()` thread-local + `[hipDNN-forwarded]` error-string prefix.
7. Document the explicit non-coverage of env-var / logging mapping for this phase. Forwarded ops in this phase should be ones whose behavior is *not* sensitive to MIOpen env vars or logging-related state.
8. Run the existing test suite with forwarding on (`MIOPEN_HIPDNN_FORWARDING=enabled`). Add targeted tests for any forwarded-op edge case not already covered.

Exit criteria: at least one full op family forwards to hipDNN under `MIOPEN_HIPDNN_FORWARDING=enabled` and passes the MIOpen test suite, routing tracing is wired up, and failure-mode behavior matches §6.1.

### Phase 3 — Environment variable and logging mapping

Goal: build a translation layer for the cross-cutting concerns deferred in Phase 2 — **scoped to the variables that frameworks actually use in production**, rather than an exhaustive port of the full MIOpen env-var surface. The exhaustive approach was considered and rejected: the long-tail MIOpen-only variables are largely diagnostic/tuning-debug knobs whose users are MIOpen developers (who have other means of investigation) rather than framework consumers, and trying to port all of them would blow up Phase 3's scope and timeline without delivering proportional consumer-facing value.

Tasks:
1. **Audit framework usage first.** Survey the actual MIOpen env-var usage in PyTorch ROCm, TensorFlow ROCm, ONNX Runtime, JAX/XLA, and any internal AMD-consumer build/CI to identify which MIOpen env vars are set in production environments. This is the input to the mapping prioritization — variables nobody outside MIOpen actually sets are deprioritized.
2. From the audit output, classify each variable that does see production use: (a) directly maps to a hipDNN env var, (b) maps to a hipDNN concept under a different name, (c) no hipDNN equivalent — document behavior when forwarding is on. The remaining MIOpen-specific variables (the long tail not surfaced by the audit) are documented as **incompatible with hipDNN forwarding**: when one of them is set, the wrapper either refuses to forward the affected op family or emits a one-time warning, as decided per variable.
3. Inventory logging conventions on both sides; decide whether the wrapper translates MIOpen log calls to hipDNN log calls, leaves them alone, or emits both. The routing-tracing log added in Phase 2 stays on the MIOpen side regardless — it is wrapper instrumentation, not a backend log.
4. Implement the env-var translation at wrapper init time (and on relevant per-call boundaries for variables that change kernel selection).
5. Document the mapping in user-facing docs, including the explicit list of MIOpen-only variables that are not honored when forwarding is enabled.
6. Re-run the test suite, including any tests that exercise env-var-sensitive behavior.

Exit criteria: env-var-sensitive ops that the audit (task 1) identified as production-relevant can be safely forwarded; user-facing documentation explains the mapping and lists the MIOpen-only variables that are incompatible with hipDNN forwarding.

### Phase 4 — Performance baselining

Goal: produce a definitive performance comparison between the two rollout configurations now that the env-var and logging mapping (Phase 3) is in place and the provider's direct private linkage (Phase 1) has already been validated.

The pre-wrapper baseline is **not** re-compared here. Wrapper-on-vs-pre-wrapper was the Phase 1 exit gate, and the pre-wrapper snapshot was taken on long-ago hardware/driver state — re-using it months later would mix wrapper overhead with unrelated drift in the rest of the stack. From Phase 2 onward, **`disabled` is the comparison baseline** (see §8 Testing Plan for the rationale).

![Architecture in steady state (provider direct-linked to MIOpen Private from Phase 1)](reference-images/miopen-shim-phase-4.png)

Tasks:
1. Run the broader benchmark suite (MIOpenDriver, framework-level workloads) in the two runtime configurations against the same wrapper-built binary: `MIOPEN_HIPDNN_FORWARDING=disabled` (the today-equivalent reference) and `MIOPEN_HIPDNN_FORWARDING=enabled` with hipDNN forwarding for the validated set from Phase 2. Report deltas. The reference is `disabled` measured on the same build and the same hardware/driver as `enabled`, not the pre-wrapper snapshot.
2. Re-verify the Phase 1 provider direct private linkage regression bar (§4.5) against the current build: `dlopen` of the MIOpen provider plugin succeeds and `LD_DEBUG=bindings` shows zero plugin → `libMIOpen.so.1` bindings on the MIOpen API surface. This is a regression sentinel — the direct linkage lived from Phase 1, but Phase 2/3 changes to the wrapper or to the `_impl` header could break it.
3. Decide, based on the data, whether to flip the `MIOPEN_HIPDNN_FORWARDING` default from `disabled` to `enabled` (with the validated op set). The decision criteria are out of scope for this RFC — see §9.

Exit criteria: performance numbers published for both runtime configurations (`disabled` and `enabled`); provider direct private linkage regression bar still holds; informed default-flip decision delegated to the successor RFC named in §9.

## 8. Testing Plan

The existing MIOpen test suites are the acceptance gate. There is only one shipped artifact — the wrapper build — so CI runs the suite against that artifact with the relevant `MIOPEN_HIPDNN_FORWARDING` setting per phase.

- **Unit and gtest layer** (`test/gtest/`) — every test that passes today must pass with `MIOPEN_HIPDNN_FORWARDING=disabled` (the default), no exceptions. Any divergence is a bug in the wrapper or in the routing policy, not an excuse to skip a test. From Phase 2 onward, the suite is additionally run with `MIOPEN_HIPDNN_FORWARDING=enabled` (which exercises whatever ops the Phase 2 compile-time forwarding set contains).
- **Legacy Boost.Test layer** (`test/`) — same rule.
- **MIOpenDriver** — the standalone driver is run with both env-var settings on representative shapes for each forwarded op (from Phase 2 onward).
- **CTest aggregate** (`make check`) — must be green in the default mode before merging each phase, and additionally in the enabled mode for any phase that introduces or modifies a forwarded op.

### 8.1 Perf comparison baseline policy

The prior draft included a wrapper-off vs wrapper-on A/B comparison built into the testing plan as the perf regression detector. That comparison is **dropped**: with the runtime flag, there is no wrapper-off artifact to compare against on a per-merge basis. The replacement is a two-stage baseline policy:

1. **Phase 1: pre-wrapper baseline snapshot vs `disabled`.** A snapshot of the pre-wrapper `libMIOpen.so` is taken once on the CI hardware *before* the Phase 1 split lands. The Phase 1 exit-criteria 2 measurement compares `disabled` against that snapshot to demonstrate that the wrapper hop is in the noise. The same snapshot also feeds the `abidiff` check (§7 Phase 1 task 8). **This is the only point at which the pre-wrapper baseline is the comparison reference.**
2. **Phase 2 onward: `disabled` is the comparison reference.** Once Phase 1 has established that wrapper-on (`disabled`) is performance-equivalent to the pre-wrapper baseline, the snapshot stops being useful as a comparison target. Other MIOpen work is continuing in parallel during this rollout, so a perf measurement taken weeks or months after the snapshot was captured will mix wrapper effects with unrelated drift in the rest of the stack. From Phase 2 onward, `MIOPEN_HIPDNN_FORWARDING=enabled` is measured against `MIOPEN_HIPDNN_FORWARDING=disabled` **on the same build, same hardware, same driver, same day** — the only valid attribution for "what did forwarding change."

This two-stage policy is sufficient for the rollout decisions and avoids the maintenance cost of a second build configuration in CI.

New tests added by this work:

- A "wrapper disabled" smoke test that asserts, with `MIOPEN_HIPDNN_FORWARDING=disabled`, that a call to `miopenConvolutionForward` produces the same output as the pre-wrapper baseline within tolerance. (Phase 1.)
- A "wrapper routes to hipDNN" test that asserts, for each forwarded op, the wrapper's hipDNN path returns the same result as the wrapper's Private path within tolerance. (Phase 2 onward.)
- A microbenchmark target measuring wrapper overhead on a no-op-equivalent entry point. (Phase 1.)
- A consumer-build smoke test that compiles a small program against the installed `MIOpen` public CMake target and links against `libMIOpen.so` to verify the public symbol set is unchanged and that no `_impl` declarations are reachable. (Phase 1 — guards against header leakage from §6.)
- A provider-link smoke test that compiles a small program against the installed `MIOpen_private` CMake target, includes `<miopen/miopen_impl.h>`, and verifies that the resulting binary resolves `_impl` symbols against `libMIOpen_private.so` with zero unresolved references. (Phase 1.)
- Tracing-level tests verifying that hipDNN-originated calls reach MIOpen Private directly without traversing the wrapper (the §4.5 regression bar). (Phase 1, re-verified in Phase 4.)

### 8.2 Leveraging PyTorch tests and benchmarks

PyTorch is the largest in-tree consumer of MIOpen and exercises the C API along paths that MIOpen's own test suite does not — real model graphs, autograd-driven shape combinations, mixed-dtype workloads, and the MIOpen tuning DB integration. Because the wrapper preserves the public ABI, PyTorch's own coverage is a high-value, no-extra-implementation source of integration testing.

- **PyTorch unit tests** — run the convnet, batchnorm, and RNN test modules from PyTorch's test suite (e.g. `test_nn.py`, `test_cudnn.py` paths that map to MIOpen on ROCm) against the wrapper build with the default env-var setting (`disabled`); from Phase 2 onward additionally with `MIOPEN_HIPDNN_FORWARDING=enabled` and the validated op allow-list. These catch behavioral regressions that MIOpen-only tests miss.
- **TorchBench / model zoo** — run a handful of representative workloads (ResNet, BERT, a small transformer training step) under both env-var settings. These provide a real-world signal on overhead and on any subtle numerical regressions introduced by forwarded ops.
- **PyTorch microbenchmarks** — `torch.utils.benchmark` op-level scripts on the ops being forwarded in Phase 2, to spot per-call overhead before it shows up in end-to-end numbers.

Standing up this PyTorch coverage in CI is part of Phase 1 (for the pass-through validation) and is reused throughout Phase 2–4. Coordination with the PyTorch-on-ROCm team is needed to identify the right test subsets and to ensure the wrapper-built MIOpen is exercised in their regression CI.

## 9. Future Considerations

- **Default flip.** Whether and when `MIOPEN_HIPDNN_FORWARDING` defaults to `enabled` (with some validated op set) depends entirely on Phase 4 measurements. Even after a flip, the `disabled` mode stays supported indefinitely. The actual *bar* for performing the flip — what perf/regression/coverage criteria must be hit, who signs off, what notice period downstream consumers get — is **out of scope for this RFC and will be the subject of a successor RFC** so that the decision criteria get the design attention they deserve rather than being inferred from this document. Three non-negotiables for the eventual flip, called out here so the implementation lands with the necessary affordances:
  - **Communicate loudly.** Release notes, ROCm SDK changelog, framework-team direct outreach, and a deprecation-style banner on first wrapper process startup (when the resolved mode is `enabled` by default rather than by explicit opt-in) for at least one release cycle. The Phase 1 configuration banner (§7 Phase 1 task 7) provides the hook.
  - **Documented disable path.** A consumer must be able to opt out of the new default in one step at runtime, without rebuilding. Three layered opt-outs: (a) `MIOPEN_HIPDNN_FORWARDING=disabled` at the process level, which forces pure pass-through and never goes away; (b) `MIOPEN_DISABLE_HIPDNN_FOR=*` for the "wrapper is enabled but I want every op back on MIOpen" case; (c) per-op `MIOPEN_DISABLE_HIPDNN_FOR=<op>` for surgical opt-outs. All three are picked up on next process start; no rebuild or library replacement required.
  - **Reversible by env var alone.** The runtime-flag design makes the flip itself a one-line change in the env-var parser, and it makes a *rollback* (if the flipped default causes regressions in the field) a same-day operation rather than a re-release of the library.
- **Removing the wrapper later.** If hipDNN ever subsumes MIOpen entirely, the wrapper could be retired in favor of consumers calling hipDNN directly. The phased approach keeps that door open.
- **Static-linking story.** Consumers who static-link MIOpen are out of scope here. If they become important, we would need a separate mechanism — likely link-time symbol substitution rather than runtime dispatch.
- **Windows / ROCm-on-Windows.** Direct linkage works the same on both platforms (modulo SONAME / DLL naming), so the design itself is not Windows-specific. Not blocking for the Linux rollout but worth flagging.

We deliberately do **not** plan a third routing-heuristics layer (per shape / dtype / arch) on top of the wrapper. hipDNN already has its own routing heuristics, and when hipDNN can't handle a problem it is expected to fall through to its MIOpen provider — which has MIOpen's existing heuristics. Adding a third decision layer at the wrapper level would mostly duplicate work already happening one level down. The wrapper's routing policy stays minimal: hand-curated allow/deny per entry point, driven by what hipDNN supports.

## 10. Glossary

- **MIOpen Public** — the new wrapper shared library. Exports the original public API symbol set. Ships as `libMIOpen.so` (the filename consumers link against today). Always built.
- **MIOpen Private** — the existing MIOpen library, built with its public-API entry points renamed in source to a `_impl` suffix, so the wrapper can call them without symbol collision. Ships as `libMIOpen_private.so`. Always built.
- **`_impl` suffix** — the rename applied in source to each public-API entry point inside MIOpen Private (e.g. `miopenConvolutionForward` → `miopenConvolutionForward_impl`).
- **`<miopen/miopen_impl.h>`** — the parallel `_impl`-form public header (§4.6) that declares the API under its `_impl` names. Installed alongside `<miopen/miopen.h>` but exposed only via `MIOpen_private`'s `INTERFACE_INCLUDE_DIRECTORIES`. Used by the rewired MIOpen provider plugin.
- **`MIOPEN_HIPDNN_FORWARDING`** — runtime env var controlling wrapper behavior (§4.1). Values: `disabled` (default, every call to MIOpen Private) and `enabled` (per-op routing policy consulted).
- **`MIOPEN_DISABLE_HIPDNN_FOR`** — opt-out env var (§4.4) listing op families that must not be forwarded even when `MIOPEN_HIPDNN_FORWARDING=enabled`. `=*` disables forwarding for every op. This is the documented runtime rollback knob.
- **Compile-time forwarding set** — the list, owned by MIOpen developers in the routing policy module, of ops that get forwarded to hipDNN when `MIOPEN_HIPDNN_FORWARDING=enabled`. There is no opt-in env var for individual ops; the compile-time list is the only way an op enters the forwarding set, and `MIOPEN_DISABLE_HIPDNN_FOR` is the only way to remove one at runtime.
- **Routing policy** — the centralized decision of, per entry point and per call, whether to dispatch to MIOpen Private or to hipDNN. Consulted only when `MIOPEN_HIPDNN_FORWARDING=enabled`. Inputs: the compile-time forwarding set, `MIOPEN_DISABLE_HIPDNN_FOR`, and optionally argument shape.
- **MIOpen provider** — hipDNN's existing engine plugin that calls into MIOpen. From Phase 1 onward, it uses **direct private linkage**: it links `libMIOpen_private.so` directly and calls `_impl` symbols rather than going through the wrapper.
- **Direct private linkage (provider)** — the Phase 1 wiring change (§4.5) that has the hipDNN MIOpen-provider plugin link and call MIOpen's private API directly, structurally preventing the wrapper → hipDNN → wrapper loop hazard.
- **Pass-through / forwarding** — shorthand for the wrapper's two functional modes. Pass-through: every call lands on MIOpen Private. Forwarding: selected ops are routed to hipDNN per the routing policy.
