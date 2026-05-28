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

This RFC proposes a thin wrapper layer in front of MIOpen that preserves the existing MIOpen C API verbatim from the consumer's point of view, while internally choosing — on a per-entry-point basis — whether to forward the call to the original MIOpen implementation or to hipDNN. The work is gated behind a single CMake feature flag. When the flag is **off**, MIOpen builds, links, and runs **bit-for-bit as it does today** — no symbol changes, no extra dispatch, no new dependency. When the flag is **on**, MIOpen is split into two artifacts:

- **MIOpen Private** — the existing MIOpen implementation, with the public C entry points internally renamed to a `_impl` suffix (e.g. `miopenConvolutionForward_impl`). Functionally identical to today's library, just relocated behind a private symbol surface.
- **MIOpen Public** — a new lightweight shared library that re-exports the original public symbol names (e.g. `miopenConvolutionForward`) and dispatches each call either to MIOpen Private or to hipDNN.

Two implementation options are evaluated for how the wrapper reaches the private library (see §4.3): a **public/private split with direct linkage** (preferred, pending investigation) and a **`dlopen`/`dlsym`-based wrapper that the user `LD_PRELOAD`s** (fallback). Because the public API does not change, this work is invisible to existing MIOpen consumers; they continue to call `miopenConvolutionForward` and either link against `libMIOpen.so` as today (Option A) or pick up the wrapper at process start via `LD_PRELOAD` (Option B). The forwarding decision happens behind that boundary.

### Investigation status

A first-cut Phase 1 prototype has been built and exercised; raw findings are catalogued in `0001a_InvestigationReference.md` (referred to throughout this RFC as "the investigation reference"). At a high level, the prototype confirms that the public/private split builds, links, runs, and is performance-neutral; surfaces one wrapper-induced test breakage with a clear fix path; and exposes a real gap in the Phase 4 short-circuit wiring that this RFC now treats as a design issue rather than an implementation detail. Specific findings are cited inline below.

### Phased approach

The work is broken into four phases (full detail in §7):

1. **Pass-through wrapper.** Establish the two-library split and dispatch plumbing; every entry point still forwards to MIOpen Private. Validate that overhead is negligible.
2. **Selective hipDNN forwarding.** Pick a small, low-risk set of ops; route them to hipDNN under an opt-in env var.
3. **Env var and logging mapping.** Translate MIOpen's debug/tuning/logging surface to hipDNN's, scoped to the variables that frameworks actually use in production.
4. **Provider short-circuit and baselining.** Rewire hipDNN's MIOpen provider to call MIOpen Private directly (avoiding a wrapper-back-into-hipDNN loop) and publish end-to-end performance numbers.

> **Open question — wrapper ownership.** Long-term ownership of the wrapper layer (the new `MIOpen Public` artifact, the routing policy module, and the env-var translation work in Phase 3) is not yet assigned. Candidates include an integration team or the FDE function. Needs alignment with @BradPepersAMD before Phase 1 exit so that on-call rotations, bug triage, and the eventual default-flip decision (§9) have a clear DRI.

## 2. Problem Statement

hipDNN is being positioned as the next-generation graph-execution surface for AMD's deep-learning stack, but MIOpen still has a large installed base and a stable C API that frameworks (PyTorch, TensorFlow, ONNX Runtime, JAX/XLA, etc.) depend on directly. We want a path for MIOpen consumers to benefit from hipDNN's engine ecosystem — performance work, new architectures, fusion graphs — without requiring those consumers to migrate their integration code or rebuild against a new header.

This wrapper is an explicitly **temporary** measure. The longer-term goal is for frameworks to consume hipDNN features directly so that we no longer need to add new functionality to MIOpen. Eventually MIOpen itself will be deprecated and the public API exposed by this wrapper will go away — though that endpoint is expected to be a long way off, is out of scope for this RFC, and is not blocked or driven by the work proposed here. In the meantime, the wrapper gives us a way to put framework calls in front of hipDNN without forcing framework teams to do migration work first.

Concretely, we want the ability to:

- Redirect specific MIOpen API calls to hipDNN at runtime, on a per-entry-point basis.
- Roll the redirection out incrementally — one operation, or one path within an operation, at a time — and roll it back just as easily.
- Prove that the forwarding layer itself adds negligible overhead before we migrate any production traffic onto it.
- Keep the "no wrapper" build path completely untouched, so that the wrapper feature cannot regress consumers who haven't opted in. The flag-off build must produce the exact same artifact MIOpen produces today.
- Eventually let hipDNN's existing MIOpen provider bypass the wrapper to avoid the round-trip cost when hipDNN is already in the call stack.

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

### 4.1 Build-time feature flag

A new CMake option, tentatively `MIOPEN_ENABLE_HIPDNN_WRAPPER` (name TBD; default: `OFF`), controls the entire mechanism.

- **OFF**: The build is byte-for-byte equivalent to the current build. `libMIOpen.so` exports the existing public symbols, and those symbols resolve to today's implementations. No new files are compiled in, no new libraries are produced, no dependency on hipDNN is introduced.
- **ON**: The build splits MIOpen into two artifacts and adds a runtime dependency on hipDNN. Details below.

The flag must guard *everything* — source files, install rules, `target_link_libraries` entries, generated headers — so that flipping it cannot leak forwarding code into the default build.

### 4.2 Symbol surface split (architecture)

When the flag is on, the public API symbols are split across two libraries — a thin **Public** wrapper and the existing implementation, which becomes **Private**. After Phase 1, the wrapper is a pure pass-through to Private; from Phase 2 onward the wrapper may also forward to hipDNN, and Phase 4 short-circuits the hipDNN provider directly into Private (see §4.5 and the diagram in §7 Phase 1 for the pass-through state, and the diagram in §7 Phase 4 for the post-Phase-4 steady state).

- **MIOpen Private**: contains the entire current MIOpen library, with each public-API entry point internally renamed from `miopenFoo` to `miopenFoo_impl`. The renaming is applied uniformly via a code-generation step or a header-level macro (`#define miopenFoo miopenFoo_impl` when building Private), so that the .cpp source files do not need to be edited individually. All other MIOpen symbols (internal helpers, solver classes, etc.) are unchanged.
- **MIOpen Public**: a new, small shared library. For each public entry point, it defines a wrapper that decides where to dispatch and forwards the arguments. The exported symbol set is identical to today's `libMIOpen.so`, so no consumer code change is required.

How the two libraries are physically packaged, named on disk, and bound together is the subject of §4.3.

### 4.3 Implementation options

Two implementation options are under evaluation. **Option A is preferred but needs investigation** to confirm feasibility (build-system mechanics, symbol visibility, packaging implications). Option B is the fallback if A turns out to be impractical.

#### 4.3.1 Option A — Public/private split with direct linkage  *(preferred)*

The existing MIOpen `.so` filename is **reused for the public wrapper**. The implementation library is renamed.

| Aspect | Value |
|---|---|
| Public wrapper artifact | `libMIOpen.so` (same filename consumers link against today) |
| Private implementation artifact | `libMIOpen_private.so` |
| Symbol rename in Private | each public entry point → `_impl` suffix |
| Wrapper → Private binding | direct link at build time (`-lMIOpen_private`) |
| Wrapper → hipDNN binding | direct link at build time (`-lhipdnn`) |
| Consumer impact | none — same `-lMIOpen` link line, same SONAME |
| User runtime opt-in | none — picked up automatically when wrapper-on builds are deployed |

Why preferred: the call path is a plain function call (one indirection at most), the wrapper is a normal library that can be packaged and shipped through the existing channels, and consumers don't need to know the wrapper exists. It also makes the Phase 4 short-circuit straightforward — hipDNN's MIOpen provider just changes its link line to `-lMIOpen_private`.

Investigation status (see investigation reference §1, §2, §4 for full data):
- **Confirmed viable.** The prototype produces `libMIOpen.so.1` as the wrapper and `libMIOpen_private.so.1` as the implementation, with the wrapper holding 263 public exports and the private library holding 263 `_impl` symbols (4 non-renamed survivors, none implicated in any collision). `MIOpenDriver` links and runs correctly against the split; gtest results show 261/265 binaries clean. The header-level rename mechanism worked without per-file edits (gated by `MIOPEN_BUILDING_PRIVATE`; see §4.6 for the mechanism trade-off).
- **Symbol-rename collisions.** No collisions observed in the prototype against the renamed surface; the symbol count check from §6 row 2 has so far come back clean.
- **Still to be validated.**
  - SONAME / SOVERSION inheritance and `abidiff` equivalence of the wrapper-on `libMIOpen.so` against wrapper-off. The prototype builds reuse `.so.1`; the formal CI gates listed in §6 row 8 and §8 are not yet wired up.
  - Packaging implications for ROCm distributors: shipping a second `.so` and ensuring both end up in the right paths.
    - **Coordinator:** TBD (named owner required before Phase 1 exit). Stakeholders to contact before Phase 1 exit: **TheRock superbuild maintainers, ROCm release-engineering, distro packagers (Ubuntu, RHEL, SUSE), container-image maintainers (`rocm/dev-*` images), Conda-forge ROCm, and the framework wheel teams (PyTorch ROCm, TensorFlow ROCm)**. Sign-off must be captured in a tracking ticket. Each of these consumers has a build/packaging assumption that "MIOpen ships as one `.so`"; surfacing this late turns into a Phase 4 release-readiness blocker.
  - Interaction with the existing `MIOpen_with_plugins` CMake target. The plugin-enabled build composes additional artifacts into the MIOpen install tree and must continue to work — or be explicitly marked unsupported — when the wrapper flag is on. Not covered by the current investigation; remains open.

#### 4.3.2 Option B — `dlopen`/`dlsym` wrapper, `LD_PRELOAD`-based  *(fallback)*

Today's `libMIOpen.so` is **left completely unchanged**. The wrapper ships as a separate library that the user `LD_PRELOAD`s.

| Aspect | Value |
|---|---|
| Existing MIOpen artifact | `libMIOpen.so` — **unchanged** |
| New wrapper artifact | `libMIOpen_wrapper.so` (name TBD) — separate `.so`, not part of the default link line |
| Wrapper → MIOpen binding | `dlopen("libMIOpen.so")` + `dlsym("miopenFoo")` at process start |
| Wrapper → hipDNN binding | `dlopen("libhipdnn.so")` + `dlsym("hipdnnFoo")` at process start |
| Consumer impact | must set `LD_PRELOAD=libMIOpen_wrapper.so` at process launch to opt in |
| User runtime opt-in | explicit, per-process, via `LD_PRELOAD` |

Why this is the fallback: the user-facing opt-in is more invasive (`LD_PRELOAD` is fragile, awkward in container/CI environments, and easy to forget), and the call path adds a function-pointer indirection plus all the usual `dlopen` lifecycle concerns. It does, however, leave the existing MIOpen build entirely alone and decouples the wrapper from the hipDNN ABI at link time, so it is the safer option if Option A turns out to require build-system changes we cannot stomach.

#### 4.3.3 Findings from the `dlopen`/`dlsym` investigation

An investigation was conducted into the dynamic-loader-based mechanism that Option B depends on. The findings below catalog the problems that surfaced; together they are the reason Option B is treated as a fallback only, to be used solely if Option A is found infeasible. Several of these findings would also need to be addressed *in addition to* the Option A work if we ever fell back to B, materially expanding its scope beyond what the §4.3.2 table suggests.

**Correctness and lifecycle**

- **Symbol resolution ordering.** `dlsym(RTLD_NEXT, ...)` and `dlopen` of an explicit path behave differently in subtle ways. Getting the choice wrong means the wrapper either fails to find the implementation symbol or recurses into itself; both failure modes are silent until the first forwarded call.
- **Initialization order.** Wrapper init touches logging, env-var parsing, and hipDNN handle creation. In some load orders this happens before MIOpen's own static initializers run. The failure mode is a deadlock or crash at process init, which is hostile to diagnose because it happens before any user code executes.
- **Fork/exec safety.** `dlopen`'d handles and cached function pointers must survive `fork()` without re-initialization. HIP runtime state interacts badly with `fork()` in general, and the wrapper would inherit those interactions on top of its own dynamic-loader state.
- **Thread-safety of first-call resolution.** The standard "cache the function pointer on first call" pattern requires either a one-shot init (e.g. `pthread_once`) or an explicit memory barrier. Without one, concurrent first calls can observe a torn read of the pointer. This is the kind of bug that passes CI on a quiet machine and surfaces only under production concurrency.

**Operational**

- **No link-time validation of the hipDNN ABI.** The §4.3.2 table frames this as "degrades gracefully," but in practice an ABI break shows up as `dlsym` returning `NULL` deep inside a workload rather than as a build failure. The failure mode moves from CI to production, which is the wrong direction.
- **Debuggability.** Stack traces go through function pointers, so breakpoints set on `miopenConvolutionForward` do not fire where users expect. Static analysis tools (`nm`, `ldd`, `readelf -d`) do not reveal the real dependency graph because hipDNN is loaded at runtime. This raises the on-call and support burden for every issue that touches the wrapper.
- **Library discovery is environment-dependent.** `dlopen("libhipdnn.so")` resolves via `LD_LIBRARY_PATH`, `ld.so.cache`, and RPATH at runtime. Picking up the wrong hipDNN — an older system copy, a sibling install, a leftover from a different ROCm version — is silent and hard to diagnose. Option A eliminates this entirely by binding both libraries at link time.

**Portability**

- **POSIX-only API.** `dlopen`/`dlsym` is POSIX. Windows requires `LoadLibrary`/`GetProcAddress`, which means the resolver code is forked by platform and the wrapper needs a portability shim for any work that needs to run on ROCm-on-Windows. Option A's direct linkage works identically on both platforms (modulo SONAME / DLL naming).

**Performance**

- **Per-call indirection.** Every forwarded call goes through a function pointer, which defeats inlining at the call site and is harder to argue is "in the noise" than a plain function call. This is the same overhead concern flagged in §6 row 1 as "non-negligible … especially under Option B's `dlopen`/`dlsym` indirection," but the investigation confirms the indirection is structural to Option B — it cannot be optimized away while keeping the dynamic-loader mechanism.

Collectively, these findings convert the §4.3.2 "Option B is the fallback if A turns out to be impractical" framing into a stronger statement: Option B carries real correctness, operational, and portability costs that Option A does not, so Option A should be made to work unless a hard blocker is found. The previously open Option B edge cases (`RTLD_DEEPBIND` interactions, static-linked consumers where `LD_PRELOAD` does not interpose, `setuid` binaries where the loader strips `LD_PRELOAD`) are folded into the operational findings above and would each need explicit handling if Option B were ever adopted.

#### 4.3.4 Findings from the Option A prototype

A prototype Option A build was stood up; full data is in the investigation reference (§1–§4). Headlines:

- **Build and link.** The flag-on configuration produces the expected `libMIOpen.so.1` wrapper + `libMIOpen_private.so.1` implementation. `MIOpenDriver` and the gtest binaries link and run; smoke tests show identical solver selection and verification numerics between flag-off and flag-on (investigation reference §1).
- **Runtime path proof.** With per-stub `fprintf` instrumentation, a single `MIOpenDriver convfp16` invocation produced 45 wrapper-traversal lines on flag-on and 0 on flag-off — every public call in the run went through the wrapper hop, no hidden bypasses (investigation reference §2).
- **Test suite.** 261 of 265 gtest binaries clean on flag-on. Of the 4 case failures: 2 are wrapper-induced (`test_smoke_tuning_policy` asserts on captured `MIOPEN_LOG_FUNCTION` stderr that now contains `_impl` names) and 2 are environmental (`test_db_sync` missing a `.kdb` file). 2 timeouts (`test_lrn`, `test_soft_max`) were progressing when the per-binary 1200 s budget fired — not failures (investigation reference §3). The wrapper-induced fix is tracked as a Phase 1 follow-up (see §7 Phase 1 task list).
- **Performance.** Aggregate GPU kernel time differs by ≤2% between configurations (signed direction varies by phase — i.e., noise); wall-clock is the same or slightly faster on flag-on. Direct microbenchmark of the worst-case no-work API (`miopenGetVersion`) puts the wrapper hop at **≈1 ns/call (upper bound)** in steady state; cold-load cost is **+6 major page faults** and ≤1 MB RSS (investigation reference §4). The negligible-overhead premise that justifies the whole design is empirically supported.

Taken together, the Option A prototype clears the feasibility bar for Phase 1. Outstanding items (SONAME/SOVERSION CI gates, packaging coordination, `MIOpen_with_plugins` interaction) remain as noted in §4.3.1, and the Phase 4 short-circuit needs a design adjustment that §4.5 now addresses.

### 4.4 Per-entry-point routing policy

Each wrapper function consults a routing decision: "should this call go to hipDNN or to MIOpen Private?" In Phase 1 the answer is always "Private". As later phases add hipDNN coverage, the policy becomes more nuanced.

Routing inputs that the policy may consider:

- The entry point itself (some are forwarded, some are not).
- Argument shape (e.g. forward only certain layouts/dtypes).
- An opt-in environment variable (e.g. `MIOPEN_USE_HIPDNN_FOR=convolution,batchnorm`) so that the routing set can be changed without rebuilding.
- A companion opt-out environment variable (`MIOPEN_DISABLE_HIPDNN_FOR=...`) so that a single op family can be force-routed back to MIOpen Private without rebuilding or having to enumerate the full enable-list. This is the recovery lever when a forwarded op misbehaves in production and is the documented rollback path called out in §9 (the "loud" default-flip communication promises this knob will exist).
- A compile-time list for entry points that are known-good and always forwarded.

When both `MIOPEN_USE_HIPDNN_FOR` and `MIOPEN_DISABLE_HIPDNN_FOR` are set, `MIOPEN_DISABLE_HIPDNN_FOR` wins so that operators always have a definitive kill switch.

The routing decision is centralized in one place (a small `routing.cpp` or equivalent) so that adding/removing a forwarded op is a one-line change.

### 4.5 Phase 4: hipDNN MIOpen-provider short-circuit

Today, hipDNN's MIOpen provider plugin calls into `libMIOpen.so`. Once the wrapper exists, that call would land in MIOpen Public, which might in turn forward back into hipDNN — a loop that, even if guarded against actual recursion, adds an unnecessary hop.

In Phase 4 the MIOpen provider is changed to link directly against `libMIOpen_private.so` and call the `_impl` symbols. This bypasses the wrapper entirely when hipDNN is the original caller, so the wrapper only sits in the path when an *external* consumer (e.g. PyTorch) called MIOpen.

This change is contained to the provider and does not affect other consumers.

**Gap finding from the investigation.** The provider's CMakeLists already conditionally links `MIOpen_private` when that target is exported, and the prototype confirms the link selection works as intended — but the resulting plugin **links and then fails to load** because the provider's translation units `#include <miopen/miopen.h>` and therefore reference the unrenamed names (`miopenCreate`), while `libMIOpen_private.so.1` only exports `_impl`-suffixed names (`miopenCreate_impl`). The `dlopen` step aborts at the plugin's first MIOpen relocation. Additionally, `libMIOpen.so.1` carries a hard `DT_NEEDED` on `libMIOpen_private.so.1`, so a flag-on install always loads both files transitively — the short-circuit changes *which exports bind*, not *which files load*. Full evidence in investigation reference §5.

The Phase 4 design therefore requires a consumer-side rename mechanism. The investigation reference §5.7 sketches three options:

1. **Install the rename header** and have `find_package(miopen)` wire `MIOpen_private`'s `INTERFACE_INCLUDE_DIRECTORIES` / `INTERFACE_COMPILE_OPTIONS` to apply `-DMIOPEN_BUILDING_PRIVATE -include miopen_private_rename.h` automatically. Lowest friction for plugin authors; matches the assumption the existing CMake snippet was written against.
2. **Ship a parallel public header** declaring the API in `_impl` form; the provider conditionally includes that header when targeting `MIOpen_private`. Avoids macro rewriting at the cost of conditional compilation in consumer code.
3. **Keep the rename header private, export aliases** so `libMIOpen_private.so.1` publishes both the renamed and unrenamed names. No consumer-side work, but re-creates the risk Phase 4 was trying to remove (the private library becoming reachable via the public name surface).

Selection of the remediation is deferred — an upcoming pivot to the broader plan may change the constraints that drive it. Until then, all three remain on the table and Phase 4 cannot be declared done. The regression bar is concrete: `dlopen` of a plugin built against `MIOpen_private` must succeed, and `LD_DEBUG=bindings` must show the plugin's MIOpen calls binding to `_impl` symbols in `libMIOpen_private.so.1` with zero bindings into `libMIOpen.so.1`.

### 4.6 Header story

The public header `miopen.h` is unchanged. Internally, when building MIOpen Private, an additional generated header (or a small block of `#define`s in `config.h`) renames the declared functions to their `_impl` variants. Consumers never see this header and never see the `_impl` names.

**Rename mechanism — `#define` vs. localized macro push/pop.** Two approaches are on the table; the choice is open pending the upcoming plan pivot.

- **Simple `#define` via forced include** (used by the prototype). The private build applies `-DMIOPEN_BUILDING_PRIVATE -include miopen_private_rename.h`, and the header `#define`s each `miopenFoo` to `miopenFoo_impl` for the whole TU. This is what the investigation reference §1, §2, §5 exercised and it produced a clean rename (263/263 entry points on the wrapper side, 263 `_impl` symbols on the private side, only 4 unrelated non-renamed survivors). The two theoretical fragility concerns are: (a) any third-party header included in the same TU that mentions a `miopenFoo` symbol *as a string* (debug-tracing macros, logging facility metadata, embedded format strings) gets silently rewritten; (b) the forced-include order matters, so any TU that subverts it can pick up unrenamed signatures and confusing link errors. Neither has surfaced in the prototype, but the surface area is broad enough that the concern is not dismissed outright.
- **Localized `#pragma push_macro` / `pop_macro`** (alternative). The public header's declarations are wrapped with `#pragma push_macro("miopenFoo")` / `#define miopenFoo miopenFoo_impl` / declaration / `#pragma pop_macro("miopenFoo")` so the rename only takes effect across the declaration itself and is reverted before any subsequent include. Avoids both fragility modes above at the cost of mechanically heavier header machinery.

The prototype evidence supports the simpler approach being workable in practice; the heavier mechanism remains the safer long-term option if either fragility mode ever surfaces. The pick will be finalized as part of the plan pivot, and the consumer-side mirror of whatever is chosen feeds directly into the Phase 4 remediation discussion in §4.5.

**Guarantee: the rename header must not be included by any header in the public include path.** Including it transitively into a header that consumers (or MIOpen Public's wrapper TUs) include would cause the rename to apply inconsistently — renamed in Private TUs (good) and unrenamed elsewhere (silent ambiguity about whether the caller is reaching Public's wrapper or Private's `_impl`). This invariant is enforced by CI checks documented in §6 row 8 and §8.

Additionally, MIOpen internal headers under `projects/miopen/src/include/miopen/*.hpp` should be audited for inline or template helpers that legitimately reference public symbols (rare but possible — typically for debug or utility paths). Any such helper would get the rename applied inconsistently between Public and Private TUs if it is included by both; the audit either confirms none exist or moves the call site behind a `_impl`-aware indirection.

## 5. Key Design Decisions

Each decision below ties back to one or more bullets in the problem statement (§2). If a decision can't be tied back, that's a signal either to drop the decision or to widen the problem statement.

### 5.1 Split into Public + Private libraries instead of a single library with two symbol sets

**Ties to:** "Roll the redirection out incrementally … and roll it back just as easily" and "Eventually let hipDNN's existing MIOpen provider bypass the wrapper."

A single shared object that exports both `miopenConvolutionForward` and `miopenConvolutionForward_impl` would also work, but a split keeps the dependency direction one-way, which is what the incremental-rollout and provider-bypass goals depend on.

| | Pros | Cons |
|---|---|---|
| **Split (chosen)** | Explicit one-way dependency: Public depends on Private; Private has no knowledge of the wrapper. hipDNN's provider can link Private directly (Phase 4). Wrapper symbol footprint stays small and inspectable. Wrapper can be stripped/optimized independently of MIOpen proper. | Two artifacts to package and ship. Slightly more build-system complexity. |
| **Single `.so` with both symbol sets** | One artifact. No SONAME juggling. | The `_impl` symbols leak into anything that links against MIOpen — the provider can't cleanly bypass the wrapper without seeing both surfaces. Harder to reason about which symbol set is "the API." |

### 5.2 Build-time flag instead of always shipping the wrapper

**Ties to:** "Keep the 'no wrapper' build path completely untouched, so that the wrapper feature cannot regress consumers who haven't opted in."

A runtime flag ("wrapper present but always dispatches to Private") would not satisfy that bullet — the wrapper would still be in the call path and in the binary, so the flag-off artifact would no longer be byte-equivalent to today's MIOpen.

| | Pros | Cons |
|---|---|---|
| **Build-time flag (chosen)** | Flag-off artifact is bit-identical to today's MIOpen. Clean A/B comparison for perf regression detection. Wrapper code can be developed in-tree without affecting the shipped binary. | Two build configurations to maintain in CI. Concretely: worst-case this doubles MIOpen build + test wall-clock in any pipeline that exercises both configurations end-to-end. Order of magnitude on current CI hardware, based on observed run times — the most recent successful MIOpen CI build was ~6 hours, and TheRock Release portable Linux package builds have been ranging from ~4 to ~6.5 hours — that's an extra **~4–6.5 hours per pipeline run** for a full-arch build and test cycle if the wrapper-on configuration is serialized after wrapper-off (to be refined once we have measured numbers on the Phase 1 prototype). Mitigation: run wrapper-on as a separate, parallelizable matrix job rather than serializing after wrapper-off (which keeps wall-clock roughly flat at the cost of ~2× compute), and limit the wrapper-on matrix to a representative arch subset for routine pre-merge runs while running the full matrix nightly. |
| **Runtime flag** | Single artifact; one build to test. | Wrapper is always present in the binary and in the call path, so flag-off is no longer equivalent to today's MIOpen — the very thing the problem statement requires us to preserve. |

### 5.3 Prefer Option A (direct linkage) over Option B (`dlopen`/`dlsym` + `LD_PRELOAD`)

**Ties to:** "Prove that the forwarding layer itself adds negligible overhead" and "Eventually let hipDNN's existing MIOpen provider bypass the wrapper."

| | Pros | Cons |
|---|---|---|
| **Option A — direct linkage (preferred)** | No user-side opt-in (no `LD_PRELOAD`). Plain function-call dispatch — easiest to argue is overhead-free. Provider short-circuit in Phase 4 is a one-line link-line change. SONAME is preserved, so consumers see no change. | Requires build-system changes to produce `libMIOpen.so` as a wrapper while renaming the implementation. Investigation required before commitment. |
| **Option B — `dlopen`/`dlsym` + `LD_PRELOAD`** | Existing MIOpen build is untouched. Wrapper is decoupled from hipDNN ABI at link time; degrades gracefully if hipDNN is missing. | `LD_PRELOAD` opt-in is fragile in container/CI environments and easy to forget. Function-pointer indirection plus `dlopen` lifecycle concerns (init ordering, fork/exec, first-call thread-safety). ABI breaks surface in production rather than at link time. Library discovery depends on `LD_LIBRARY_PATH`/RPATH/`ld.so.cache` — wrong-hipDNN pickup is silent. Resolver is POSIX-only; Windows needs a separate `LoadLibrary` shim. Provider short-circuit in Phase 4 is awkward — the provider would need its own opt-out from the preload. See §4.3.3 for the full investigation findings. |

The preference is Option A precisely because the negligible-overhead and provider-bypass goals are easier to reason about with a plain function call than with a preload-based shim. The §4.3.3 investigation strengthens this preference on the Option B side; the §4.3.4 prototype clears the feasibility bar on the Option A side (build, link, run, gtest, performance all confirmed). Option B remains the fallback only if outstanding Option A items (SONAME/SOVERSION CI gates, packaging coordination, `MIOpen_with_plugins`) surface blocking issues.

### 5.4 Phase env vars and logging separately

**Ties to:** "Roll the redirection out incrementally — one operation, or one path within an operation, at a time."

MIOpen has a substantial set of debug / tuning / logging environment variables and a logging subsystem with its own conventions. hipDNN has a different set. Mapping these requires care — some MIOpen knobs have no hipDNN analog and vice versa, some affect the call path before any forwarding decision can be made, and some affect kernel selection inside MIOpen Private in ways the wrapper can't influence. Gating the forwarding work on resolving the mapping would block incremental rollout. Phase 2 explicitly defers it (by only forwarding ops insensitive to env vars), Phase 3 takes it on as the focus.

| | Pros | Cons |
|---|---|---|
| **Phase separately (chosen)** | Forwarding rollout starts immediately on insensitive ops. Env-var mapping gets the design attention it deserves rather than being rushed. | A subset of ops is off-limits for forwarding until Phase 3 lands. |
| **Block on env-var/logging mapping first** | Single, complete forwarding story at launch. | Pushes any user-visible benefit out by months for a problem that doesn't affect early targets. |

### 5.5 Short-circuit the MIOpen provider in Phase 4 rather than Phase 1

**Ties to:** "Eventually let hipDNN's existing MIOpen provider bypass the wrapper" and "Roll the redirection out incrementally."

| | Pros | Cons |
|---|---|---|
| **Phase 4 (chosen)** | Phase 1 changes stay minimal. The wrapper is validated end-to-end before hipDNN's build is touched. | Until Phase 4, hipDNN-originated calls go through the wrapper unnecessarily — but the wrapper is a no-op pass-through during that window, so the cost is the indirection only. |
| **Phase 1** | Provider short-circuit available from day one. | Couples the wrapper rollout to a hipDNN build change; expands the blast radius of any Phase 1 issue. |

## 6. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Per-call overhead from the wrapper is non-negligible for short-running ops (especially under Option B's `dlopen`/`dlsym` indirection) | Low (Option A) / Medium (Option B) | High | **Largely mitigated for Option A by the prototype**: microbenchmark of the worst-case no-work API (`miopenGetVersion`) shows ≈1 ns/call upper-bound steady-state cost and +6 major page faults at cold load (investigation reference §4). For any real op that touches the GPU, wrapper hop is unmeasurable. Under Option B the function-pointer indirection remains structural — see §4.3.3. |
| Renaming public symbols to `_impl` collides with existing internal symbols somewhere in MIOpen | Low | Medium | The renaming is mechanical and applied via the public header; collisions surface at link time. We grep the source for `_impl` suffixes on the public-API names before flipping the flag. Prototype run produced 263/263 renamed entry points with 4 unrelated non-renamed survivors and no observed collisions (investigation reference §5.4). |
| Behavioral divergence between MIOpen and hipDNN for a forwarded op (precision, edge cases, error codes) | Medium | High | Phase 2 forwards a small, well-tested set first. The existing MIOpen test suite is the acceptance gate — anything that regresses against MIOpen-only behavior is reverted. Prototype runs at the pass-through level confirm 261/265 gtest binaries clean and identical solver selection in smoke tests (investigation reference §1, §3), so the gate is in working order before any forwarding lands. |
| Environment variables silently change meaning when the wrapper is on | High if unaddressed | Medium | Phase 2 explicitly does *not* forward any op whose behavior is sensitive to env vars we haven't mapped. Phase 3 builds the mapping. |
| hipDNN ABI churn breaks the wrapper | Medium | Medium | Under Option A, breakage surfaces at link time and is caught in CI. Under Option B, version-check at load time and degrade to "dispatch to Private" if hipDNN is missing or incompatible. Either way, hipDNN ABI changes need to be coordinated with the wrapper's release cadence. |
| Consumers that statically link MIOpen lose the wrapper indirection | Low–Medium (assumption — needs confirmation) | Low | Currently flagged as low-likelihood, but this is an assumption that needs confirmation: **before Phase 1 exit, reach out to the framework and downstream-consumer teams (PyTorch ROCm, TensorFlow ROCm, internal AMD consumers) to inventory whether anyone actually static-links MIOpen today.** If any production consumer does, the risk likelihood and the mitigation both need to be re-scoped. Until then, document that the wrapper requires the shared-library build; if a static-link consumer surfaces and matters, a separate mechanism (likely link-time symbol substitution) is needed. |
| Loop / recursion if the MIOpen provider in hipDNN re-enters the wrapper before Phase 4 | Low | High | Until Phase 2, the wrapper never forwards to hipDNN, so the loop cannot occur. From Phase 2 onward, the routing policy explicitly excludes calls originating from the MIOpen provider (detectable via a thread-local guard). Phase 4 makes this structural by hooking the provider directly to Private. |
| Symbol-rename macros leak into consumer builds via a transitively-included header | Low | High | The `_impl` rename is confined to a header that is only included when *building* MIOpen Private. Public-installed headers do not include it. Verified by CI checks: (a) compile a sample consumer against the installed headers; (b) `grep -r '_impl' <staged-public-headers>` must return zero matches; (c) `nm` the compiled sample-consumer binary and assert no `*_impl` symbols appear in its symbol table; (d) `abidiff` the wrapper-on `libMIOpen.so` against the wrapper-off `libMIOpen.so` and assert the exported public symbol set is identical. See §4.6 for the alternative header mechanism (`#pragma push_macro` / `pop_macro`) that limits the rename to a localized scope rather than file-global, which is recommended over the simple `#define` approach. |
| hipDNN ↔ MIOpen runtime version mismatch under Option B | Medium (Option B only) | Medium | Option B resolves both libraries via `dlopen`/`dlsym` at process start, so the wrapper picks up whatever `libMIOpen.so` and `libhipdnn.so` the loader finds — which may not match the versions the wrapper was built against. Mitigation: wrapper performs a version handshake at init (reads `miopenGetVersion` and the hipDNN equivalent) and refuses to enable hipDNN forwarding if either is outside a documented compatibility window, falling back to pass-through-to-Private with a one-time warning to stderr. Option A does not have this problem because both libraries are link-time bound. |
| hipDNN is missing, broken, or fails at runtime | Medium | High | Behavior is defined per failure class (load-time vs per-call) with a thread-local "last forwarded error" mechanism so on-call can tell which side failed. See §6.1 immediately below. |

### 6.1 Failure-mode behavior for hipDNN forwarding

Three failure classes, each with explicit defined behavior:

- **Load-time failure** — `libhipdnn.so` cannot be loaded at process start (Option B) or the wrapper's hipDNN dependency cannot be resolved at link time (Option A). The wrapper logs once to stderr at init, sets an internal `hipdnn_disabled` flag, and routes every call to MIOpen Private for the lifetime of the process. No retries; no per-call `dlopen` storms.
- **Per-call hipDNN failure** — a forwarding attempt returns a hipDNN-side error. **The failure is propagated to the caller as a translated `miopenStatus_t`. We do not silently fall back to MIOpen Private mid-execution**, because silent fallback would mask correctness regressions (hipDNN returned a wrong-but-not-error result, MIOpen would have returned right) and hide perf regressions. Operators retain `MIOPEN_DISABLE_HIPDNN_FOR=<op>` (§4.4) as the kill switch for a flaking op.
- **Error-code distinguishability.** A translated `miopenStatus_t` alone is ambiguous — on-call cannot tell from `MIOPEN_STATUS_INTERNAL_ERROR` whether to read MIOpen source or hipDNN source. The wrapper provides:
  - a thread-local accessor `miopenGetLastForwardedError()` returning the hipDNN-side status (if any) for the most recent forwarded call on the current thread, and
  - a `[hipDNN-forwarded]` prefix on the `miopenGetErrorString` message for any forwarded-error code, so log scraping is unambiguous.
  - A new dedicated error code (`MIOPEN_STATUS_FORWARDED_ERROR`) was considered but rejected for Phase 2 because it would change the visible status set for any consumer that opts into forwarding; the thread-local accessor is additive and does not. To be re-evaluated if forwarded-error patterns turn out to warrant their own bucket.

## 7. Execution Plan

The work is broken into four phases. Each phase ends with the existing MIOpen test suite green in both wrapper-off and wrapper-on configurations.

Estimates below are gross — measured in **person-sprints** (one person-sprint = one engineer for a two-week sprint). They assume Option A (direct linkage); Option B would shift roughly one sprint of effort from build-system work into runtime/`dlopen` plumbing in Phase 1, with the rest of the plan unchanged. Total: **~8–14 person-sprints**.

### Phase 1 — Lightweight pass-through wrapper

**Estimate: 2–3 person-sprints.** A prototype covering tasks 2–6 below has been built and characterized; remaining Phase 1 work is the CI plumbing (tasks 1, 7), the PyTorch coverage (task 8), and a wrapper-induced test fix surfaced by task 9. Findings are in the investigation reference; specific subsections cited below.

Goal: establish the two-library split and the dispatch plumbing, with **all** entry points dispatching to MIOpen Private. From the consumer's perspective, behavior is identical to today.

![Architecture after Phase 1 (pass-through)](reference-images/miopen-shim-phase-1.png)

Tasks:
1. Add the `MIOPEN_ENABLE_HIPDNN_WRAPPER` CMake option (default OFF) and verify the OFF build produces a byte-equivalent `libMIOpen.so`. CMake should emit a clear status message at configure time announcing the wrapper state ("MIOpen wrapper: OFF — building today's MIOpen unchanged" vs. "MIOpen wrapper: ON — Phase N forwarding scope: …") so that downstream packagers and CI consumers cannot miss the configuration change. *(Status: the flag exists and toggles the build as expected; byte-equivalent `abidiff` check is part of task 7 and not yet wired up.)*
2. Investigate Option A feasibility (build-system mechanics, SONAME and SOVERSION inheritance, packaging). If blocking issues are found, fall back to Option B and amend the RFC. *(Status: build-system mechanics and runtime path confirmed by the prototype — investigation reference §1, §4. SONAME/SOVERSION CI gate and packaging coordination still open; see §4.3.1.)*
3. Introduce the symbol-rename header that, when building MIOpen Private, redefines each public entry point to its `_impl` variant. *(Status: prototype uses `-DMIOPEN_BUILDING_PRIVATE -include miopen_private_rename.h` and produced 263/263 renamed entry points. Final mechanism choice — simple `-include` vs. localized `pragma push_macro`/`pop_macro` — remains open per §4.6.)*
4. Add the new MIOpen Private build target (the existing MIOpen library with the rename applied) and the new wrapper target that becomes `libMIOpen.so`. Under Option A, the wrapper directly links MIOpen Private and hipDNN; SONAME and SOVERSION are inherited from today's `libMIOpen.so`. Under Option B, the wrapper resolves both via `dlopen`/`dlsym` at init. *(Status: targets exist in the prototype; SONAME/SOVERSION inheritance verification still pending the CI gate in task 7.)*
5. Generate the wrapper source file. Each entry point has a stub that forwards the call to the corresponding `_impl` symbol (a plain function call under Option A; a cached function-pointer dispatch under Option B). *(Status: 263 stubs in place; runtime-traversal proof in investigation reference §2.)*
6. Microbenchmark wrapper overhead on a representative short-running op (e.g. small `miopenSetTensor`). Confirm overhead is in the noise. *(Status: done. Worst-case no-work API (`miopenGetVersion`) measured at ≈1 ns/call upper-bound steady state, +6 major page faults cold-load, ≤1 MB RSS — investigation reference §4. End-to-end conv benchmark shows aggregate GPU time differs by ≤2% with signed direction varying by phase, i.e. noise.)*
7. Add the **header-leakage CI checks** described in §6 row 8 and §4.6: (a) `grep -r '_impl' <staged public headers>` returns no matches; (b) sample-consumer build links against installed `libMIOpen.so` and `nm` of the resulting binary shows no `_impl` symbols; (c) `abidiff` compares wrapper-on `libMIOpen.so` against wrapper-off and asserts the exported public symbol set is identical. *(Status: not yet wired into CI.)*
8. **Stand up the PyTorch-on-ROCm CI coverage** described in §8.1 (the convnet/batchnorm/RNN test modules and the TorchBench subset) against the wrapper-on build. This is the highest-coverage realistic-workload signal for the pass-through validation and is reused throughout Phases 2–4. Requires coordination with the PyTorch-on-ROCm team to identify the right test subsets and to get the wrapper-on build exercised in their regression pipelines; this coordination starts now so it is in place before Phase 2 forwards anything. *(Status: not yet started.)*
9. Run both the MIOpen test suite **and** the MIOpen provider tests (in `dnn-providers/miopen-provider/`) in both wrapper-off and wrapper-on configurations; all must be green. *(Status: gtest pass — 261/265 binaries clean on wrapper-on, with 2 wrapper-induced case failures in `test_smoke_tuning_policy` (see task 10), 2 environmental case failures in `test_db_sync` (missing `gfx942.kdb`), and 2 timeouts in `test_lrn`/`test_soft_max` that were still progressing at the 1200 s per-binary budget — investigation reference §3. Provider tests: §5 of the investigation reference uncovered the Phase 4 wiring gap discussed in §4.5; the gap surfaces at Phase 4 (when the provider is rewired against `MIOpen_private`) and does not block Phase 1's pass-through pathway.)*
10. **Wrapper-induced test fix** (known-issue follow-up surfaced by task 9). `test_smoke_tuning_policy` asserts on captured stderr from `MIOPEN_LOG_FUNCTION`, which now contains `_impl`-suffixed function names rather than the public names the test was written against. Two fixes are under consideration: (a) normalize the wrapper's `_impl` log output back to the public name, or (b) relax the test assertion to accept the `_impl` form. Pick is TBD; both are low-risk and either resolves the failure. Investigation reference §3 has the failure details.

Exit criteria (Phase 1 — pass-through):

1. **Functional parity.** Wrapper-on build passes the same MIOpen test suite + MIOpen provider tests + sample-consumer build smoke test that wrapper-off does. Phase 1 explicitly does not exercise the env-var or logging mapping — the forwarded op set is empty, so env-var-sensitive paths are out of scope for this phase. Tasks deliberately select ops whose pass-through behavior is insensitive to env-var state. Env-var coverage is the focus of Phase 3. *(Status: provisionally met for the gtest suite — 261/265 binaries clean — pending resolution of the wrapper-induced `test_smoke_tuning_policy` failure (task 10) and the environmental `test_db_sync` setup gap. Investigation reference §3.)*
2. **Steady-state wall-clock overhead.** Wrapper adds < 1% wall-clock overhead on a representative end-to-end workload (target metric to be confirmed in Phase 1 after seeing real numbers). This is the gate for the wrapper itself being acceptable as a permanent piece of the call path. *(Status: met. End-to-end conv workload shows wall-clock the same or slightly faster on wrapper-on; aggregate GPU kernel time deltas ≤ 2% with signed direction varying by phase. Investigation reference §4.)*
3. **Per-call overhead breakdown (gate, not a target).** Measure overhead from each of: parameter translation, hipDNN graph building (where applicable in later phases — measured here so the Phase 2 number can be compared cleanly), dispatch indirection. If any single component dominates the < 1% budget, escalate before Phase 2 starts. Mitigation strategies that are pre-approved if translation overhead is a problem: caching translated graphs, caching descriptor conversions, lazy materialization of hipDNN handles. *(Status: dispatch indirection measured at ≈1 ns/call upper bound. Parameter translation and hipDNN graph building are Phase 2 measurements — there is no forwarded op to translate against in Phase 1.)*
4. **Cold-start vs warm-start compilation time.** Measure first-call latency separately from steady-state. The risk we are gating against is that hipRTC compilation (or any other one-time JIT work on either side) happens *twice* — once on a MIOpen Private path and once on a hipDNN path — for a workload that, under wrapper-off, would have compiled it only once. The exit criterion is that cold-start latency for any forwarded op family on wrapper-on is ≤ 1.10× cold-start under wrapper-off, and warm-start is within the < 1% steady-state budget above. *(Status: pass-through cold-load characterized — +6 major page faults, ≤1 MB RSS, no detectable wall-clock change. The double-compilation gate stays open until Phase 2 introduces a real hipDNN forwarding path. Investigation reference §4.)*
5. **PyTorch-on-ROCm CI signal.** The PyTorch coverage stood up in task 8 is green against the wrapper-on build (pass-through configuration) for the agreed test subset. *(Status: not yet stood up.)*
6. **Header-leakage CI gates green.** The three checks added in task 7 are wired into CI and pass. *(Status: not yet wired up.)*

### Phase 2 — hipDNN forwarding for selected entry points

**Estimate: 3–5 person-sprints** (scales with the number of op families in the initial set; assume one to two).

Goal: actually forward a small, low-risk set of API calls to hipDNN. Defer environment-variable and logging concerns.

Tasks:
1. Pick the initial forwarding set. Candidates are entry points where hipDNN already has full coverage and the behavior is well-understood (likely starting with a single op family).
2. Implement the routing policy module — a single source file that the wrapper consults to decide Private vs. hipDNN per call.
3. Implement the hipDNN call paths: argument translation from MIOpen descriptors to hipDNN graph + variant pack, hipDNN execution, result translation back to `miopenStatus_t`.
4. Add the opt-in env var (`MIOPEN_USE_HIPDNN_FOR=...`) and the opt-out env var (`MIOPEN_DISABLE_HIPDNN_FOR=...`, §4.4) for runtime selection of which ops are forwarded. Disable wins over enable.
5. **Add routing tracing.** When `MIOPEN_LOG_LEVEL` (or a wrapper-specific equivalent) is set, every wrapper invocation logs a single line containing the entry-point name, the routing decision (Private vs. hipDNN), and the reason (compile-time forced, env-var opt-in, env-var opt-out, policy fall-through). This is the on-call instrument for "why did this call go where I didn't expect" investigations and is a prerequisite for confidently flipping `MIOPEN_ENABLE_HIPDNN_WRAPPER` defaults in any environment.
6. Implement the failure-mode behavior defined in §6.1: load-time fallback to Private with a one-time warning, per-call hipDNN errors propagated (not silently absorbed), and the `miopenGetLastForwardedError()` thread-local + `[hipDNN-forwarded]` error-string prefix.
7. Document the explicit non-coverage of env-var / logging mapping for this phase. Forwarded ops in this phase should be ones whose behavior is *not* sensitive to MIOpen env vars or logging-related state.
8. Run the existing test suite with forwarding on for the selected ops. Add targeted tests for any forwarded-op edge case not already covered.

Exit criteria: at least one full op family forwards to hipDNN under the env-var opt-in and passes the MIOpen test suite, routing tracing is wired up, and failure-mode behavior matches §6.1.

### Phase 3 — Environment variable and logging mapping

**Estimate: 2–4 person-sprints.**

Goal: build a translation layer for the cross-cutting concerns deferred in Phase 2 — **scoped to the variables that frameworks actually use in production**, rather than an exhaustive port of the full MIOpen env-var surface. The exhaustive approach was considered and rejected: the long-tail MIOpen-only variables are largely diagnostic/tuning-debug knobs whose users are MIOpen developers (who have other means of investigation) rather than framework consumers, and trying to port all of them would blow up Phase 3's scope and timeline without delivering proportional consumer-facing value.

Tasks:
1. **Audit framework usage first.** Survey the actual MIOpen env-var usage in PyTorch ROCm, TensorFlow ROCm, ONNX Runtime, JAX/XLA, and any internal AMD-consumer build/CI to identify which MIOpen env vars are set in production environments. This is the input to the mapping prioritization — variables nobody outside MIOpen actually sets are deprioritized.
2. From the audit output, classify each variable that does see production use: (a) directly maps to a hipDNN env var, (b) maps to a hipDNN concept under a different name, (c) no hipDNN equivalent — document behavior when forwarding is on. The remaining MIOpen-specific variables (the long tail not surfaced by the audit) are documented as **incompatible with hipDNN forwarding**: when one of them is set, the wrapper either refuses to forward the affected op family or emits a one-time warning, as decided per variable.
3. Inventory logging conventions on both sides; decide whether the wrapper translates MIOpen log calls to hipDNN log calls, leaves them alone, or emits both. The routing-tracing log added in Phase 2 stays on the MIOpen side regardless — it is wrapper instrumentation, not a backend log.
4. Implement the env-var translation at wrapper init time (and on relevant per-call boundaries for variables that change kernel selection).
5. Document the mapping in user-facing docs, including the explicit list of MIOpen-only variables that are not honored when forwarding is enabled.
6. Re-run the test suite, including any tests that exercise env-var-sensitive behavior.

Exit criteria: env-var-sensitive ops that the audit (task 1) identified as production-relevant can be safely forwarded; user-facing documentation explains the mapping and lists the MIOpen-only variables that are incompatible with hipDNN forwarding.

### Phase 4 — MIOpen-provider short-circuit and performance baselining

**Estimate: 1–2 person-sprints.**

Goal: avoid the wrapper hop when hipDNN is the original caller, and produce a definitive performance comparison.

![Architecture after Phase 4 (steady state)](reference-images/miopen-shim-phase-4.png)

Tasks:
1. **Pick the consumer-side rename mechanism.** The provider's existing `find_package(miopen)` + conditional `MIOpen_private` link snippet is necessary but not sufficient: the prototype confirms it links the correct library but the plugin fails at `dlopen` because consumer TUs call unrenamed names while `libMIOpen_private.so.1` only exports `_impl` names (full evidence in investigation reference §5; design discussion in §4.5). Pick one of the three options sketched in §4.5 (install the rename header + auto-apply via `MIOpen_private`'s INTERFACE properties; ship a parallel `_impl`-form public header; export private-side aliases). The choice is deferred pending an upcoming plan pivot and is a hard prerequisite for the remaining Phase 4 tasks.
2. Modify hipDNN's MIOpen provider (`dnn-providers/miopen-provider/`) to link against MIOpen Private and call `_impl` symbols directly, using whatever rename mechanism was picked in task 1.
3. Verify (with a tracing test) that hipDNN → MIOpen-provider → MIOpen calls do not pass through the wrapper. Concrete bar from investigation reference §5.7: `dlopen` of a `MIOpen_private`-linked plugin must succeed, and `LD_DEBUG=bindings` must show plugin → `libMIOpen_private.so.1` bindings on `_impl` symbols with **zero** bindings into `libMIOpen.so.1`.
4. Run the broader benchmark suite (MIOpenDriver, framework-level workloads) in three configurations: wrapper-off (baseline), wrapper-on with no hipDNN forwarding, wrapper-on with hipDNN forwarding for the supported set. Report deltas.
5. Decide, based on the data, whether to flip the wrapper default to ON for any downstream consumer.

Exit criteria: consumer-side rename mechanism is in place and the regression bar in task 3 holds; performance numbers published; informed decision on wrapper default.

## 8. Testing Plan

The existing MIOpen test suites are the acceptance gate. They run in both wrapper configurations from Phase 1 onward.

- **Unit and gtest layer** (`test/gtest/`) — every test that passes today must pass with the wrapper on, no exceptions. Any divergence is a bug in the wrapper or in the routing policy, not an excuse to skip a test.
- **Legacy Boost.Test layer** (`test/`) — same rule.
- **MIOpenDriver** — the standalone driver is run against both configurations on representative shapes for each forwarded op.
- **CTest aggregate** (`make check`) — must be green in both configurations before merging each phase.

New tests added by this work:

- A "wrapper passthrough" smoke test that asserts, with the wrapper on but routing to Private, that a call to `miopenConvolutionForward` produces the same output as the wrapper-off build. (Phase 1.)
- A "wrapper routes to hipDNN" test that asserts, for each forwarded op, the wrapper's hipDNN path returns the same result as the wrapper's Private path within tolerance. (Phase 2 onward.)
- A microbenchmark target measuring wrapper overhead on a no-op-equivalent entry point. (Phase 1.)
- A consumer-build smoke test that compiles a small program against the installed MIOpen headers and links against `libMIOpen.so` to verify the public symbol set is unchanged. (Phase 1 — guards against header leakage from §6.)
- Tracing-level tests verifying that hipDNN-originated calls bypass the wrapper. (Phase 4.)

All tests are run in CI for both wrapper-off and wrapper-on builds. Wrapper-off remains the default until Phase 4 concludes otherwise.

### 8.1 Leveraging PyTorch tests and benchmarks

PyTorch is the largest in-tree consumer of MIOpen and exercises the C API along paths that MIOpen's own test suite does not — real model graphs, autograd-driven shape combinations, mixed-dtype workloads, and the MIOpen tuning DB integration. Because the wrapper preserves the public ABI, PyTorch's own coverage is a high-value, no-extra-implementation source of integration testing.

- **PyTorch unit tests** — run the convnet, batchnorm, and RNN test modules from PyTorch's test suite (e.g. `test_nn.py`, `test_cudnn.py` paths that map to MIOpen on ROCm) against both wrapper-off and wrapper-on builds. These catch behavioral regressions that MIOpen-only tests miss.
- **TorchBench / model zoo** — run a handful of representative workloads (ResNet, BERT, a small transformer training step) under both configurations. These provide a real-world signal on overhead and on any subtle numerical regressions introduced by forwarded ops.
- **PyTorch microbenchmarks** — `torch.utils.benchmark` op-level scripts on the ops being forwarded in Phase 2, to spot per-call overhead before it shows up in end-to-end numbers.

Standing up this PyTorch coverage in CI is part of Phase 1 (for the pass-through validation) and is reused throughout Phase 2–4. Coordination with the PyTorch-on-ROCm team is needed to identify the right test subsets and to ensure the wrapper-on build is exercised in their regression CI.

## 9. Future Considerations

- **Default flip.** Whether and when `MIOPEN_ENABLE_HIPDNN_WRAPPER` defaults to ON depends entirely on Phase 4 measurements. Even if we flip the default, the OFF path stays supported. The actual *bar* for performing the flip — what perf/regression/coverage criteria must be hit, who signs off, what notice period downstream consumers get — is **out of scope for this RFC and will be the subject of a successor RFC** so that the decision criteria get the design attention they deserve rather than being inferred from this document. Two non-negotiables for the eventual flip, called out here so the implementation lands with the necessary affordances:
  - **Communicate loudly.** Release notes, ROCm SDK changelog, framework-team direct outreach, prominent CMake configure-time status messages on the new default, and a deprecation-style banner on first wrapper-on process startup for at least one release cycle.
  - **Documented disable path.** A consumer must be able to opt out of the new default in one step. Three layered opt-outs: (a) the build-time `MIOPEN_ENABLE_HIPDNN_WRAPPER=OFF` flag never goes away; (b) `MIOPEN_DISABLE_HIPDNN_FOR=*` at runtime turns off every forwarded op in the wrapper-on build; (c) per-op `MIOPEN_DISABLE_HIPDNN_FOR=<op>` for surgical opt-outs.
- **Removing the wrapper later.** If hipDNN ever subsumes MIOpen entirely, the wrapper could be retired in favor of consumers calling hipDNN directly. The phased approach keeps that door open.
- **Static-linking story.** Consumers who static-link MIOpen are out of scope here. If they become important, we would need a separate mechanism — likely link-time symbol substitution rather than runtime dispatch.
- **Windows / ROCm-on-Windows.** Under Option B, `dlopen`/`dlsym` is POSIX-only and the Windows equivalent is `LoadLibrary`/`GetProcAddress`; the wrapper would need a thin abstraction. Under Option A this concern largely goes away (direct linkage works the same on both platforms, modulo SONAME / DLL naming). Not blocking for the Linux rollout but worth flagging.

We deliberately do **not** plan a third routing-heuristics layer (per shape / dtype / arch) on top of the wrapper. hipDNN already has its own routing heuristics, and when hipDNN can't handle a problem it is expected to fall through to its MIOpen provider — which has MIOpen's existing heuristics. Adding a third decision layer at the wrapper level would mostly duplicate work already happening one level down. The wrapper's routing policy stays minimal: hand-curated allow/deny per entry point, driven by what hipDNN supports.

## 10. Glossary

- **MIOpen Public** — the new wrapper shared library. Exports the original public API symbol set. Under Option A, ships as `libMIOpen.so` (replacing the existing artifact in that role). Under Option B, ships as a separate `libMIOpen_wrapper.so` that the user `LD_PRELOAD`s.
- **MIOpen Private** — the existing MIOpen library, built with its public-API entry points renamed to a `_impl` suffix, so the wrapper can call them without symbol collision. Under Option A, ships as `libMIOpen_private.so`. Under Option B, the existing `libMIOpen.so` is left untouched and the wrapper resolves its symbols at runtime.
- **`_impl` suffix** — the rename applied to each public-API entry point inside MIOpen Private (e.g. `miopenConvolutionForward` → `miopenConvolutionForward_impl`).
- **Option A / Option B** — the two implementation options under evaluation in §4.3. Option A: public/private split with direct linkage (preferred). Option B: `dlopen`/`dlsym`-based wrapper that the user `LD_PRELOAD`s (fallback).
- **Wrapper-off / wrapper-on** — shorthand for the two states of the `MIOPEN_ENABLE_HIPDNN_WRAPPER` CMake option.
- **Routing policy** — the centralized decision of, per entry point and per call, whether to dispatch to MIOpen Private or to hipDNN.
- **MIOpen provider** — hipDNN's existing engine plugin that calls into MIOpen. In Phase 4, it is rewired to call MIOpen Private directly.
- **`dlopen` / `dlsym`** — POSIX dynamic-loader primitives used by Option B's wrapper to resolve `_impl` and hipDNN entry points at runtime, avoiding a build-time link dependency.
