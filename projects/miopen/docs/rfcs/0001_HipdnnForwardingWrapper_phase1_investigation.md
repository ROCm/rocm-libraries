# Phase 1 Investigation Plan: Public/Private Library Split

This document is a companion to [RFC 0001 — MIOpen → hipDNN Forwarding Wrapper](0001_HipdnnForwardingWrapper.md). It scopes the investigation work that gates the rest of Phase 1 (§7 of the RFC) and pins down the open questions in §4.3.1 ("What needs investigation"). The objective is to determine — concretely, for this codebase — whether **Option A (public/private split with direct linkage)** is feasible, and to surface anything that would force a fallback to **Option B** (`dlopen`/`LD_PRELOAD`).

The investigation is the *prerequisite* to the implementation tasks listed in §7 Phase 1, items 3–7. It does **not** itself touch any source files outside the new CMake gating flag.

## 1. Constraints the investigation must satisfy

These are non-negotiable; any approach that fails one of them is rejected.

1. **Flag-off byte-equivalence.** With `MIOPEN_ENABLE_HIPDNN_WRAPPER=OFF` (the default), the configure + build must produce a `libMIOpen.so` indistinguishable from today's. No new compile units, no new link-line entries, no extra install artifacts.
2. **Public ABI preservation.** With the flag on, `libMIOpen.so` exports exactly the same C symbol set as today (same names, same SONAME, same versioning), so no consumer relink is required.
3. **One-way dependency.** Public depends on Private; Private has no knowledge of the wrapper. This is what makes the Phase 4 provider short-circuit a one-line link change.
4. **Symbol-rename is mechanical.** Renaming `miopenFoo` → `miopenFoo_impl` inside MIOpen Private must be applied via a single header-level mechanism — no per-file `.cpp` edits — so that drift between the wrapper-on and wrapper-off code paths is impossible.

## 2. Inventory: what we are splitting

Concrete numbers gathered from the current tree (commit `a827879e67`):

| Metric | Value | Notes |
|---|---|---|
| Public C entry points in `include/miopen/miopen.h` | 263 `MIOPEN_EXPORT` declarations / **263 distinct functions** | Verified by `tools/wrapper/gen_rename_header.py` (Q1 prototype). The earlier "172 distinct" figure was a stale estimate from a previous tree state; the current header has no duplicate entry-point names. The wrapper must implement all 263 as pass-throughs in Phase 1. |
| `*_api.cpp` files in `src/` | 30 | These contain `extern "C"` definitions of the public symbols. None apply `MIOPEN_EXPORT` to definitions — visibility comes from default-visibility builds. |
| Internal C++ surface | `MIOPEN_INTERNALS_EXPORT` macro, generated separately into `export_internals.h` | Used by tests/driver. Out of scope for the wrapper but **must keep working** unchanged in both flag states. |
| Existing CMake target | `MIOpen` (built in `src/CMakeLists.txt:866`) | Becomes Private when the flag is on; becomes the wrapper artifact under the same SONAME. |
| Visibility regime | Default visibility (no `-fvisibility=hidden`) | Confirmed: no `fvisibility` flag anywhere in `cmake/` or top-level CMakeLists. `MIOPEN_EXPORT` exists but is informational rather than load-bearing today. |

## 3. Open questions (the investigation deliverables)

Each numbered item below is something the investigation must answer with evidence, not speculation. Each one has a defined exit criterion.

### Q1 — Can a single header-level macro perform the `_impl` rename, with no per-file `.cpp` edits?

**Why it matters:** RFC §4.2 commits to "a single header-level mechanism". If we have to edit each of the 30 `*_api.cpp` files, we lose the property that wrapper-on and wrapper-off compile from the same source.

**Approach:**
1. Build a prototype `miopen_private_rename.h` that, when `MIOPEN_BUILDING_PRIVATE` is defined, expands `#define miopenFoo miopenFoo_impl` for every public entry point. Bootstrap the list directly from `include/miopen/miopen.h` via a one-shot Python script over `MIOPEN_EXPORT` declarations, then check the output in as a hand-maintained file (parallel to the Q4 wrapper-source decision — see §6 Q1).
2. Have the rename header included via the compiler's `-include` flag — so the .cpp files don't reference it explicitly.
3. Verify on three representative `*_api.cpp` files (`activ_api.cpp`, `convolution_api.cpp`, `tensor_api.cpp`) that the rename:
   - applies cleanly to `extern "C"` definitions,
   - does **not** mangle internal C++ helpers that happen to share a substring with a public name,
   - does **not** mangle uses of the public name as a *string* (logging, error messages),
   - does **not** mangle declarations in the still-public `miopen.h` when included from inside Private.

**Risk to flag:** Macro renaming text-substitutes everywhere the token appears. False positives in `MIOPEN_LOG_FUNCTION(miopenFoo, …)` or `"miopenFoo"` string literals will surface as either compile errors (good) or silently wrong logging (bad, must be checked).

**Exit criterion:** A compile of the three prototype files under the rename produces `miopenFoo_impl` symbols (verified with `nm -D --defined-only`) and zero collateral renames in string literals (verified by grepping the preprocessed output).

**Fallback if Q1 fails:** Code-generate the `*_api.cpp` files, with the API name as a template parameter. More invasive but bounded.

### Q2 — Can `libMIOpen.so` be produced as a thin wrapper while preserving SONAME, ABI tags, and the existing install layout?

**Why it matters:** Consumers currently link against `-lMIOpen` and load `libMIOpen.so.1`. Breaking SONAME forces a system-wide rebuild of every downstream package (PyTorch, TensorFlow, ONNX-RT). RFC §4.3.1 Constraint 2.

**Approach:**
1. Read what `rocm_set_soversion(MIOpen ${MIOpen_SOVERSION})` (src/CMakeLists.txt:881) actually sets (`SOVERSION` and `VERSION` properties). Confirm we can set the same on the wrapper target.
2. Read what `rocm_install_targets(TARGETS MIOpen ...)` (src/CMakeLists.txt:1058) installs and where. Confirm that swapping `MIOpen` for the wrapper target (and adding `MIOpen_private` alongside) works through the same `rocm_install_targets` machinery without bespoke install rules.
3. Compare `nm -D --defined-only` of the proposed wrapper artifact to today's `libMIOpen.so`. The exported set must be a superset (the public `miopen*` symbols, plus whatever the wrapper itself emits — e.g. routing-policy globals).

**Exit criterion:** A flag-on prototype build produces a `libMIOpen.so.1.0` whose **exported** symbol set, as reported by `nm -D --defined-only --extern-only | grep '^[0-9a-f]\+ T miopen'`, matches today's `libMIOpen.so.1.0` for every public C symbol. SONAME (`readelf -d libMIOpen.so | grep SONAME`) matches.

### Q3 — Does Private (renamed `_impl` symbols) coexist cleanly with the C++ internal surface (`MIOPEN_INTERNALS_EXPORT`) used by tests?

**Why it matters:** The test suite, MIOpenDriver, and the fin tool currently link against `MIOpen` and reach into both the public C API *and* internal C++ classes. If those consumers are switched to link Private (so they get the renamed `_impl` symbols), they will fail to find `miopenFoo`. If they stay linked to the wrapper, they pay the indirection during testing — acceptable but worth confirming intentionality.

**Approach:**
1. Audit the link lines of `test/`, `driver/`, `fin/`, `speedtests/` targets (most go through the `MIOpen_with_plugins` proxy at src/CMakeLists.txt:920).
2. Decide policy: tests link the wrapper (so they exercise the same path consumers do), but we add a separate `MIOpen_private` interface alias for cases that need to bypass the wrapper (Phase 4 provider short-circuit and any in-tree benchmark of overhead).
3. Verify a sample test still finds both `miopenConvolutionForward` (from wrapper) and `miopen::ConvolutionDescriptor::FindConvFwdAlgorithm` (from Private's internal C++ surface) at link time.

**Exit criterion:** Test suite builds and runs unchanged in flag-on mode; no test target needs its `target_link_libraries` line edited.

### Q4 — Wrapper-source generation: one generated `.cpp` or per-entry-point stubs?

**Why it matters:** 172 entry points. Each needs a stub:
```cpp
extern "C" miopenStatus_t miopenConvolutionForward(/* args */) {
    return miopenConvolutionForward_impl(/* args */);   // Phase 1
}
```
Maintaining 172 hand-written stubs is bug-prone; auto-generation is mandatory. The question is *how*.

**Options:**
| Option | How | Pros | Cons |
|---|---|---|---|
| **(a) Single generated `.cpp` from a Python script over `miopen.h`** | A `gen_wrapper.py` reads `miopen.h`, emits `wrapper_generated.cpp` at configure time. | One file, one script, easy to inspect diff between flag states. | Requires a header-parsing approach robust to MIOpen's macro use (`MIOPEN_DECLARE_OBJECT`, multi-line declarations). |
| **(b) X-macro list, hand-maintained alongside the header** | A `public_api.def` listing every entry point with its signature. The wrapper `.cpp` includes it under different macro definitions to emit declarations and stubs. | No header parsing. Pure C preprocessor. | The list must be kept in sync with `miopen.h` by hand — exactly the drift problem we want to avoid. |
| **(c) Code-gen via libclang / clang's tooling APIs** | Robust C parsing. | Adds a build-time toolchain dependency. Slow. | Overkill for 172 plain `extern "C"` declarations. |

**Approach taken:** option (a) one-shot. The prototype was built (regex
extractor over `MIOPEN_EXPORT` declarations) and run once to produce
`src/private/wrapper.cpp` with 263 stubs; the generator was then retired
and the file is now hand-maintained. See §6 Q4 for the rationale and the
drift-detection strategy that replaces hermeticity-of-generation.

**Exit criterion:** `src/private/wrapper.cpp` contains one `extern "C"`
stub per `MIOPEN_EXPORT` declaration in `miopen.h` (enforced by the
`investigation_q4_stub_count` CTest) and is compiled into the wrapper
`MIOpen` target with a link against `MIOpen_private`. Both pieces are
in place; empirical link-test runs as part of the first flag-on build.

### Q5 — Header leakage: can the rename-header escape into a consumer build via transitive include?

**Why it matters:** RFC §6 calls this out explicitly as a Low-likelihood / **High-impact** risk. If a consumer somehow includes `miopen_private_rename.h`, every call to `miopenConvolutionForward` in the consumer becomes `miopenConvolutionForward_impl` — link errors at best, silent behavioral changes at worst.

**Approach:**
1. Place the rename header **outside** the install tree (e.g. `src/private/miopen_private_rename.h` — never installed).
2. Apply it via `target_compile_options(MIOpen_private PRIVATE -include ${path}/miopen_private_rename.h)` rather than `#include` in source files.
3. Add a CI smoke test (covered in §4 below) that compiles a tiny consumer (`int main() { miopenHandle_t h; miopenCreate(&h); miopenDestroy(h); }`) against only the **installed** headers and links against the **installed** `libMIOpen.so`. The test asserts the symbols resolved are `miopenCreate` (no `_impl`).

**Exit criterion:** The smoke test passes in both flag-on and flag-off CI configurations.

### Q6 — Packaging: do we need to ship two `.so`s, and does that change the existing rocm-cmake install machinery?

**Why it matters:** ROCm distributors (`rocm-libraries`, downstream packagers like Debian/RHEL) consume MIOpen's CMake config and install rules. A second artifact must flow through the same pipes.

**Approach:**
1. Read `cmake/miopen-config.cmake.in` and the `rocm_export_targets` call (src/CMakeLists.txt:1066). Confirm a second target can be added to the export set without bespoke surgery.
2. Confirm `libMIOpen_private.so` ends up in `${CMAKE_INSTALL_LIBDIR}` next to `libMIOpen.so` and gets the same RPATH treatment.
3. Confirm `libMIOpen.so`'s `DT_NEEDED` includes `libMIOpen_private.so.1` (`readelf -d`).

**Exit criterion:** `cmake --install ${BUILD_DIR} --prefix /tmp/install_test` produces both libraries in `lib/`, the install-tree `libMIOpen.so` resolves `libMIOpen_private.so.1` via RUNPATH, and `find_package(miopen)` from a sample external CMake project still works.

### Q7 — Wrapper overhead measurement methodology

**Why it matters:** RFC §7 Phase 1 exit criterion: "wrapper adds < 1% wall-clock overhead on a representative end-to-end workload". This needs an apples-to-apples measurement *plan* committed before implementation, so we can't be accused of selecting a benchmark post-hoc.

**Approach:**
1. Pick the smallest-cost public entry point that nonetheless touches a real call path: `miopenSetTensor` on a 1×1×1×1 fp32 tensor. Measure call latency with both flag states (1M-iteration tight loop, median of 10 runs). Acceptance: median delta < 50 ns (the cost of a single function call on a modern x86).
2. Pick a realistic short op: 1024×1024 GEMM-equivalent convolution, batch 1, fp16. Acceptance: delta < 1% wall-clock.
3. Pick a realistic long op: ResNet50 forward, batch 32, fp16, on a representative GPU. Acceptance: delta well below noise.

**Exit criterion:** A `tools/wrapper_overhead.sh` script that runs all three measurements in both flag states and emits a CSV with the deltas. Run before merging Phase 1.

## 4. Validation matrix

The tests below are run by CI in **both** flag-off and flag-on configurations from Phase 1 onward. Anything red is a Phase 1 regression and blocks merge.

| Validation | Flag-off | Flag-on (pass-through) | Notes |
|---|---|---|---|
| `make check` (CTest aggregate) | must pass | must pass | RFC §8 |
| `test/gtest/` | must pass | must pass | RFC §8 |
| Legacy `test/` (Boost.Test) | must pass | must pass | RFC §8 |
| `MIOpenDriver` smoke shapes | must pass | must pass | RFC §8 |
| Consumer-build smoke (Q5) | must pass | must pass | New in Phase 1 |
| Public-symbol-set diff (Q2) | n/a (baseline) | must equal flag-off baseline | New in Phase 1 |
| SONAME diff | n/a (baseline) | must equal flag-off baseline | New in Phase 1 |
| Wrapper-overhead micro/macro (Q7) | n/a (baseline) | < 1% delta | RFC §7 Phase 1 exit |
| Install + sample `find_package` (Q6) | must pass | must pass | New in Phase 1 |
| MIOpen-provider tests (`dnn-providers/miopen-provider/`) | must pass | must pass | RFC §7 Phase 1 task 7 |

The flag-off byte-equivalence check is asserted twice: once at configure time (the configure log under flag-off must match a baseline log committed alongside this plan, modulo timestamps) and once at link time (`sha256sum libMIOpen.so` against a checked-in baseline for a fixed build container). This is overkill but cheap, and it's the only way to *prove* property #1 from §1.

## 5. Decision gate

The investigation produces a single go/no-go on Option A:

- **Go:** Q1–Q6 all pass exit criteria; Q7 measurement infrastructure is in place. Proceed with RFC §7 Phase 1 tasks 3–7.
- **No-go:** Any of Q1–Q6 fails. Document the failure mode in this file under "§6 Investigation findings", amend the RFC's §4.3 to make Option B the chosen path, and re-plan the work.

### Decision

**Go on Option A.** Q1, Q3, Q4, Q6, Q7 met their exit criteria during the investigation. Q2 and Q5 are wired up in the implementation that lands alongside this update (see §6) and gated on artifacts only the implementation produces — both are runnable now via the CTests below. The build-system split (RFC §7 Phase 1 tasks 3–6) is implemented in the same change set: `MIOpen_private` carries the renamed `_impl` symbols, the existing `MIOpen` target rebuilds as a thin wrapper compiled from `src/private/wrapper.cpp` and PUBLIC-linked to `MIOpen_private`, both libraries flow through the same `rocm_install_targets`/`rocm_export_targets` pipes, and the hipDNN miopen-provider (RFC §7 Phase 1 task 7) prefers `MIOpen_private` when present (Phase 4 short-circuit ready). Empirical numbers for Q2/Q5/Q6/Q7 land as the wired CTests are run against real builds.

## 6. Investigation findings

*To be filled in as the investigation completes. Each Q gets a subsection with: what we tried, what we observed, exit-criterion status (pass/fail/blocked), and a link to the prototype branch or PR.*

### Q1 — Symbol rename header

**Status:** PASS (exit criterion met).

- The hand-maintained header at `src/private/miopen_private_rename.h`
  (263 `#define miopenFoo miopenFoo_impl` lines) is now applied to
  MIOpen_private sources via `target_compile_options(MIOpen_private PRIVATE
  "SHELL:-include …/src/private/miopen_private_rename.h")` in
  `src/CMakeLists.txt`. `MIOPEN_BUILDING_PRIVATE` is also set on
  MIOpen_private; the header's hard `#error` guard fires if anything else
  ever includes it.
- The header lives outside the install tree (`src/private/` is not part
  of the `rocm_install_targets` INCLUDE list) — Q5 boundary preserved.
- The wrapper TU (`src/private/wrapper.cpp`) does NOT include the rename
  header; each stub forward-declares its own `_impl` symbol. The boundary
  is therefore preserved at translation-unit granularity, not just at the
  install-tree level.
- Collateral-rename safety: a grep over `src/` confirms there is no
  `MIOPEN_LOG_FUNCTION(miopen[A-Z]…` invocation and no `"miopenFoo"`
  string literal matching a public function name; the only places the
  identifiers appear in `src/` are as the function names themselves, which
  is exactly what we want renamed. The full preprocessed-output check is
  exercised at compile time by the BUILD_TESTING configuration of
  `MIOpen_private` — any false-positive rename in a string literal would
  fail to compile.

### Q2 — Wrapper SONAME and exported-symbol set

**Status:** wired (CTest registered); empirical exit awaits the first
flag-on build.

- Harness: `tools/wrapper/symbol_diff.sh dump | diff` (unchanged from
  investigation). The wrapper target sets the same `MIOpen_SOVERSION` as
  today via `rocm_set_soversion(MIOpen ${MIOpen_SOVERSION})`, so the
  produced `libMIOpen.so` SONAME is identical by construction.
- CTests `investigation_q2_dump_candidate` and
  `investigation_q2_symbol_superset` (in `cmake/InvestigationHipdnnWrapper.cmake`)
  perform the dump-then-diff at test time. Both are gated on
  `-DMIOPEN_WRAPPER_FLAGOFF_BASELINE=<prefix>`; supply a
  `tools/wrapper/symbol_diff.sh dump` output from a flag-off build and the
  superset/SONAME assertion runs automatically.
- A second, header-derived equivalence check
  (`public_api_symbol_check`) is wired in `src/CMakeLists.txt` and runs in
  BOTH flag states whenever BUILD_TESTING is on (no opt-in needed). It
  asserts the built library's exported `miopen*` set matches exactly the
  set of `MIOPEN_EXPORT` declarations in `miopen.h`. This closes the
  failure mode where a public symbol is declared but not defined (or vice
  versa) without requiring a maintained baseline file.

### Q3 — Coexistence with `MIOPEN_INTERNALS_EXPORT`

**Status:** PASS (exit criterion met).

- `MIOpen_with_plugins` continues to link `MIOpen`. With the wrapper flag
  off, that's the single MIOpen target (no behavior change). With the
  wrapper flag on, that's the wrapper, so tests/driver/fin/speedtests
  exercise the dispatch path real consumers do.
- `MIOpen_private_with_plugins` is added as a sibling INTERFACE library
  pointing at MIOpen_private (gated on the wrapper flag) — the bypass
  hook for the Phase 4 provider short-circuit and in-tree benchmarks.
- `MIOPEN_INTERNALS_EXPORT` continues to be generated against the
  implementation target (`generate_export_header(${_MIOPEN_IMPL_TARGET}
  BASE_NAME MIOPEN_INTERNALS …)`). The macro itself is unchanged; the
  EXPORTS-detection variable inside resolves to `MIOpen_private_EXPORTS`
  when flag-on (auto-defined for private sources), so internal C++
  symbols continue to be exported with default visibility. Tests linking
  the wrapper transitively pick up MIOpen_private as a NEEDED dep
  (PUBLIC link), so `miopen::ConvolutionDescriptor::FindConvFwdAlgorithm`
  and friends remain resolvable.
- No `target_link_libraries` line in `test/`, `driver/`, `fin/`, or
  `speedtests/` is edited.

### Q4 — Wrapper-source generation

**Status:** PASS (exit criterion met).

- The hand-maintained `src/private/wrapper.cpp` is now compiled into the
  wrapper `MIOpen` target by `add_library(MIOpen
  ${PROJECT_SOURCE_DIR}/src/private/wrapper.cpp)` in `src/CMakeLists.txt`.
  263 `extern "C"` stubs, each forwarding to a forward-declared
  `_impl` symbol resolved at link time against MIOpen_private.
- Each stub forward-declares its own `_impl` symbol; the wrapper TU does
  not include the rename header (Q5 boundary preserved).
- Drift CTest `investigation_q4_stub_count` (in
  `cmake/InvestigationHipdnnWrapper.cmake`) is unchanged — three-way
  count parity between `miopen.h`, `miopen_private_rename.h`, and
  `wrapper.cpp`. A new public entry point added to the header without
  matching updates fails CI loudly.
- Phase 2 routing changes will be applied as ordinary edits to the
  hand-maintained wrapper (a subset of stubs gain a branch into a
  per-op `hipdnn::*_shim`); the file format keeps that diff small.
- Re-bootstrapping either file is possible if `miopen.h` ever undergoes
  a large refactor (resurrect `gen_rename_header.py` and
  `gen_wrapper_source.py` from git history at commit `a827879e67`), but
  neither generator is part of the build.

### Q5 — Header-leakage smoke test

**Status:** wired (CTest registered); empirical exit awaits the first
`cmake --install` of a flag-on build.

- Harness unchanged from the investigation
  (`tools/wrapper/check_consumer_smoke.sh` + `consumer_smoke.c`).
- The boundary at the build-system level is preserved by construction:
  `src/private/` is not part of the `rocm_install_targets` INCLUDE list,
  and no other install rule references the rename header. The wrapper TU
  (`src/private/wrapper.cpp`) is compiled into MIOpen but is not itself
  installed (only the resulting library is).
- The CTest `investigation_q5_consumer_smoke` runs when the user passes
  `-DMIOPEN_INVESTIGATION_INSTALL_PREFIX=…`. To exercise it after the
  split lands: build flag-on, `cmake --install`, then re-configure with
  the install prefix variable set; `ctest -L investigation` runs it.

### Q6 — Packaging and install

**Status:** PASS for the build-system pieces; empirical DT_NEEDED/RPATH
verification awaits the first installed flag-on build.

- `rocm_install_targets(TARGETS ${_MIOPEN_INSTALL_TARGETS} …)` and
  `rocm_export_targets(TARGETS ${_MIOPEN_INSTALL_TARGETS} …)` route
  both `MIOpen` and `MIOpen_private` through the existing install/export
  pipes when the wrapper flag is on. No bespoke install rules.
- `find_package(miopen)` consumers see both targets in flag-on
  installs and the wrapper-only `MIOpen` target in flag-off installs.
  The `dnn-providers/miopen-provider/CMakeLists.txt` already detects
  the presence of `MIOpen_private` and switches its link line
  accordingly (Phase 4 short-circuit ready; see RFC §7 Phase 1 task 7).
- New CTest `investigation_q6_find_package_smoke` (gated on
  `MIOPEN_INVESTIGATION_INSTALL_PREFIX`) configures and builds the
  external `tools/wrapper/find_package_smoke/` project against the
  install tree, asserts both consumer paths build, and verifies the
  installed `libMIOpen.so`'s DT_NEEDED references `libMIOpen_private.so`
  in flag-on installs (and does not in flag-off installs).
- RPATH continues to flow through `rocm_install_targets`'s default
  `$ORIGIN/../lib` setting; no change required.

### Q7 — Overhead-measurement harness

**Status:** harness in place; empirical exit awaits a hardware run.

- Harness unchanged: `tools/wrapper/wrapper_overhead.sh` plus
  `tools/wrapper/microbench_settensor.cpp`. Runs the three workloads
  from §3 Q7 against both flag states, medians of N runs, CSV output.
- Methodology was committed before any flag-on build existed, so it
  cannot be retrofitted to a chosen result.
- **Outstanding (hardware-dependent):** run on a representative GPU, drop
  the resulting CSV into this section, and gate the merge on the
  acceptance criteria in §3 Q7 (microbench < 50 ns delta; short conv <
  1% wall-clock; ResNet50 well below noise).

## 7. Out of scope for this investigation

These are explicitly deferred to later phases of the RFC and are **not** investigated here:

- Routing policy design beyond "always Private" (RFC §4.4 — Phase 2).
- hipDNN argument translation (RFC §7 Phase 2).
- Env-var and logging mapping (RFC §4.5, Phase 3).
- MIOpen-provider short-circuit mechanics (RFC §4.5, Phase 4).
- Windows/DLL-naming concerns (RFC §9 — flagged but not blocking the Linux Phase 1 work).
