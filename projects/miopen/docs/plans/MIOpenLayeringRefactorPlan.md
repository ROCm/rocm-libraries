# MIOpen Layering Refactor — Plan & Reference

**Working branch:** `users/nhanna/miopen-layering-refactor`
**Project root:** `projects/miopen/`
**Author of prior work:** brpepers (all reference PRs closed, none merged)

---

## 1. Purpose & Goals

Clean up the MIOpen codebase by organizing how the code is laid out into **clean, enforced layers** with **no symbol leakage** and **no broken layering**. A primary driver is eliminating the leaked internal symbols that are currently pervasive (the driver and tests reach directly into MIOpen internals).

Prior work was started but never completed and has gone stale. The intent here is to **start fresh** on a clean branch, using the prior PRs/branches strictly as a **reference/guide** rather than rebasing or resurrecting them.

### Target layering

| Layer | Name | Depends on | Contents / Rules |
|-------|------|------------|------------------|
| **1** | **Core Utilities** (`common_utils`) | STL / system only — **no** MIOpen, ideally no HIP/GPU | Pure utility code pulled out of MIOpen for reuse. Logging, env-var wrapper, `ford`/`par_for`, `random`, `bfloat16`, `stringutils`, `float_equal`, metaprogramming helpers, etc. Everything above may use it; it depends on nothing in MIOpen. |
| **2** | **MIOpen library** | `common_utils` | The library itself. Must **export C API** (`miopen/miopen.h`) for anything above it that consumers (at least the driver) need. Uses core utils internally. |
| **3** | **MIOpen Utilities** (`miopen_utils`) | `common_utils` + MIOpen **public API only** | Classes that sit *above* MIOpen and add functionality that doesn't belong inside it but is commonly shared by driver and tests. E.g. the **Tensor wrapper classes** (`TensorDesc` RAII around `miopenTensorDescriptor_t`), `tensor_holder`, `verify`, CPU reference impls (`cpu_conv`, `cpu_bias`, etc.). May **not** use MIOpen internal headers. |
| **4** | **Driver & Tests** | `common_utils`, `miopen_utils`, MIOpen public API | **Driver:** must use NOTHING internal to MIOpen — public C API + the two util libs only. **Tests:** same allowance, but should prefer the public API where available and only bypass the layering when deliberately testing MIOpen internals. |

Both utility libraries are **internal build-only** artifacts: not installed, not exported via `find_package(MIOpen)`, not part of the public API. They exist purely to give the build system explicit, enforced edges between `src/`, `driver/`, and `test/`.

### Settled decisions (follow these as ground rules)

1. **Package util libraries that have compiled sources as static libs; pure-header layers are `INTERFACE` (header-only).** Reason: a static archive only buys anything ("built once, reused") when there are `.cpp` translation units to compile — it gives stronger, explicitly enforced build edges between rooms. `common_utils` contains **only headers**, so it is an `INTERFACE` target (a STATIC target would need an artificial dummy source); both reference branches (#7253/#7291) built it this way. The static-lib choice therefore bites at **`miopen_utils`**, which has compiled sources (`cpu_*` reference impls) — build that one STATIC.
2. **Enforce the layering automatically in CI.** Add the grep-based layering guards (see §6 Testing) to the CI pipeline so any future change that punches through a layer boundary is rejected. Reason: a one-time cleanup silently unravels without enforcement; the guard is what keeps the layering clean over time. It must be taught to ignore the documented `src/kernels/` exceptions below.
3. **HIP/RTC headers stay in `src/kernels/`** as a documented exception (already settled by #7291). Reason: MIOpen's runtime kernel compilation (HIPRTC) registers includes as flat name/content pairs with no directory structure, so headers referenced as `<common_utils/...>` cannot be resolved at runtime. `hip_float8.hpp` and `tensor_view.hpp` therefore stay put, referenced via relative paths with explanatory comments. Don't try to "fix" this inside the layering work — a genuine fix is a separate RTC-infrastructure change.
4. **Forwarding shims: keep them for EXTERNAL consumers only; redirect ALL internal usage to the new location.** When a relocated header's *old* path is part of the installed/public surface — i.e. it is `#include`d by an installed MIOpen header or by an external consumer (e.g. ROCm/MIFin, downstream tools) — leave a thin forwarding shim at the old path (`#include <newlocation/x.hpp>`) so external builds keep resolving it. A shim is **not** a license to skip updating internal code: every `src/`, `driver/`, `test/`, `speedtests/`, `fin/` include of the moved header must be repointed to the new path. Build-only headers that are **not** externally consumed (e.g. the `test/`- and `driver/`-local utilities that move into `miopen_utils`) get **no shim** — a clean `git mv` + direct rewrite.
   - **Required implementation order (do it in this sequence):** (a) perform the move and redirect **all internal includers directly, with NO shims** — this makes the build fail loudly wherever an internal include was missed, guaranteeing internal redirection is complete; (b) once the tree builds and tests pass, add forwarding shims **only** at the old paths that external consumers still need. This ordering prevents shims from silently masking un-migrated internal usage.
   - **CI guard treatment:** each external-facing shim is a documented, whitelisted exception (it intentionally forwards to the new location). The guard should still reject *internal* code that includes the old path — internal code must use the new location, not the shim.
   - **Reason (why this reverses the earlier "prefer direct rewrites, no shims" stance):** dropping the old public paths outright breaks external consumers that build against the installed tree (this already bit `bfloat16.hpp` with MIFin). Shims are the compatibility layer for *them*; the "no shims" discipline still applies *inside* the repo so the layering stays honest and the guard stays meaningful.

### Open design question (carried over from author — and how #7291 resolved it)
- **Can HIP code live at the Core layer?** Author leaned **yes**, but with a blocker around the kernel-include mechanism. **#7291 already settled this** (commit `d49a299a607`, "Keep kernel utility headers in src/kernels with documented exceptions"): MIOpen's **runtime kernel compilation (HIPRTC)** registers include files as **flat name/content pairs with no directory structure**, so any header referenced as `<common_utils/...>` **cannot be resolved at runtime**. Therefore `hip_float8.hpp` and `tensor_view.hpp` must **stay under `src/kernels/`**, and the consumers (`common_utils/tensor_utils.hpp`, `miopen_utils/tensor_holder.hpp`) reference them via **relative paths with explanatory comments** — a documented, accepted layering exception until the RTC infrastructure supports directory-structured includes.
- **Net guidance:** ordinary HIP *host* code can move down a layer, but **any header that gets pulled into an RTC-compiled kernel must remain in `src/kernels/`**. Carry forward #7291's "documented exception" pattern rather than re-litigating this. Genuinely fixing it = a separate RTC-infrastructure change, out of scope for the layering work.

---

## 2. Where to Find the Reference Material

All reference branches are **closed and unmerged** but still exist on `origin`. **Read them, don't build on them.** They are one linear evolution, so a single branch — **#7291** — contains the whole story.

| Read | PR / branch | What it gives you |
|------|-------------|-------------------|
| **Primary** | [#7291](https://github.com/ROCm/rocm-libraries/pull/7291) `tensor_desc_wrapper_rebased` | The **complete superset** (38 commits): both util libs, expanded public C API, `TensorDesc` wrapper, GPUMem extraction, and full removal of internal includes from driver + `miopen_utils`. ⚠️ Also dragged in stray non-`projects/miopen` artifacts (e.g. `.claude/scheduled_tasks.lock`) — mine selectively, never copy the tree wholesale. |
| **Foundation** | [#7253](https://github.com/ROCm/rocm-libraries/pull/7253) `miopen_shared_utils` | Clearest written explanation of the layering contract (read its PR description) and the smallest digestible Phase-1 diff. |
| **Contrast** | [#7252](https://github.com/ROCm/rocm-libraries/pull/7252) `fix_driver_symbols` | An *alternative* design (driver-local headers instead of shared `miopen_utils`) that was **not** adopted. Useful only as an inventory of which internal symbols the driver touches. |

Everything else is redundant with #7291 and should not be re-read: the macro-replacement step (#7254 / `fix_symbol_leaks_2`) and the GPUMem extraction (`extract_gpumem`, commit `7fa9d1c3fe8`) are both already folded into #7291. All other `users/brpepers/*` branches are unrelated to layering.

### How to inspect a reference branch without disturbing the working branch
```bash
# from repo root
git fetch origin
# primary reference (the superset): read its commit-by-commit story
git log --oneline develop..origin/users/brpepers/tensor_desc_wrapper_rebased
git diff develop...origin/users/brpepers/tensor_desc_wrapper_rebased -- projects/miopen   # full diff (ignore non-miopen paths)
# foundation, in isolation
git diff develop...origin/users/brpepers/miopen_shared_utils -- projects/miopen
git show origin/users/brpepers/miopen_shared_utils:projects/miopen/common_utils/CMakeLists.txt
# the HIPRTC kernel-include decision
git show d49a299a607
```

### Key artifacts to extract from the references
- **`common_utils/` layout & CMake** (from #7253 / #7291): `common_utils/include/common_utils/*.hpp` + `common_utils/CMakeLists.txt`. Headers moved: `algorithm`, `bfloat16`, `each_args`, `float_equal`, `ford`, `functional`, `par_for`, `random`, `rank`, `reduce_common`, `returns`, `stringutils`, `type_name` (#7253), plus `errors`, `float8`, `tensor_layout`, `tuple_utils` added in #7291.
- **`miopen_utils/` layout & CMake**: `tensor_holder`, `verify`, `cpu_conv`, `cpu_bias`, `cpu_layernorm`, `cpu_reduce_util`, `fusionHost`, `gemm`, `gpu_mem`, `mloSoftmaxHost`, `network_data`, `random`, `rnn_util`, `serialize` (+ more in #7291).
- **New public C API surface** (from #7252 / #7291) — see §5.
- **`TensorDesc` RAII wrapper** (from #7291) — wrapper over `miopenTensorDescriptor_t` using exclusively the public C API.
- **Driver-local replacement headers** — `driver_env.hpp`, `driver_log.hpp` (as kept in #7291). The fuller driver-local set (`driver_errors.hpp`, `driver_tensor.hpp`, `driver_tensor_view.hpp`, `driver_ford.hpp`, `driver_utils.hpp`) is #7252's *alternative* design — reference only if you revisit that approach.

---

## 3. Current State (measured on this branch)

Baseline coupling in `projects/miopen/` today (no util libs exist yet):

- **No** `common_utils/` or `miopen_utils/` directories exist — confirmed clean start.
- **53** driver files `#include` internal `miopen/` headers. Most-used internal headers in the driver:
  - `miopen/tensor.hpp` ×32 (the big one — `TensorDescriptor` as a value type)
  - `miopen/errors.hpp` ×14 (`MIOPEN_THROW`)
  - `miopen/tensor_view_utils.hpp` ×5, `miopen/ford.hpp` ×5, `miopen/float_equal.hpp` ×5
  - `miopen/logger.hpp` ×4, `miopen/handle.hpp` ×4, `miopen/env.hpp` ×4
  - tail: `tensor_extra`, `kernel_tuning_mode`, `bfloat16`, `tensor_ops`, `stringutils`, `rnn`, `par_for`, `dropout`, `convolution`, `gemm_v2`, `find_controls`, `pooling`, `prelu/utils`, …
- **~29** driver files include test-style verification headers; **~21** test files include driver headers — the circular `driver/ ↔ test/` tangle #7253 targets.
- **27** driver files use `MIOPEN_THROW`.
- `src/kernels/` holds **~397** files; kernels are `configure_file`'d / globbed from there. Per #7291, RTC-compiled headers must stay here (see §1 resolution).
- `bfloat16.hpp` pulls `miopen/config.h` — config coupling that must be replaced with a CMake define when moved to `common_utils` (as #7253 did).

These numbers double as a **before** snapshot for measuring progress.

---

## 4. Implementation Plan

The author's recommended high-level sequence (and an explicit decision point on **incremental vs. one big jump**):

> **Recommendation:** go **incremental by phase, one PR per phase** — *not* sub-split below a phase. Each phase ships as a **single PR**, even when large. Rationale: every PR carries fixed review/CI/merge overhead, and because each phase **blocks the next** (it's a critical-path dependency chain), splitting a phase into multiple PRs multiplies that overhead on work that downstream phases are waiting on. The lever that kept #7291 from being reviewable was that it bundled **all phases at once** (38 commits, 244 files, +8956/−7801) — the fix is one-PR-*per-phase*, not many-PRs-per-phase. Keep each phase "functionally equivalent, no runtime change" and preserve bisectability with **ordered, individually-building commits within the single PR** (`git bisect` works on commits, not PRs).

> **Ticket-generation note:** each Phase below is a JIRA **epic** that maps to **one PR**; the bulleted **Work items** are the stories/tasks (and the suggested commit boundaries within that PR); **Decisions required** are decision tickets that should be resolved (or explicitly deferred) before or early in the epic; **Depends on** drives epic ordering; **Acceptance** is the epic's done-definition. Each phase's PR also lands its **own CI layering-guard checks** (kept with the changes they enforce — see each phase's guard work item). One cross-cutting **task** (symbol-leak baseline, no PR) follows Phase 5.

### Phase 1 — Introduce `common_utils` (Core Utilities)
**Depends on:** nothing (bottom layer).
**Lib type:** **INTERFACE** (header-only) — `common_utils` has no `.cpp` sources, so an INTERFACE target is correct (a STATIC target would need an artificial dummy source); both reference branches built it this way. The static-lib decision (§1 decision 1) applies to `miopen_utils`, not here.
**Invariant:** *functionally equivalent, no runtime change* — every step is a pure code-move + include-rewrite. Any step that would change numeric or control-flow behavior is a documented concession (see work item 4 and the decisions below).

**Scope — headers to move** out of `src/include/miopen/` into `common_utils/include/common_utils/`, grouped by how much decoupling each needs (counts measured 2026-06-23; re-measure before trusting):
- **Tier A (clean leaves, no internal deps):** `algorithm`, `each_args`, `float_equal` (~51 sites, high churn), `par_for`, `rank`, `returns`, `type_name`.
- **Tier B (deps are in-group, stay clean once the group moves):** `functional` (→`each_args`,`returns`), `ford` (→`par_for`,`each_args`,`returns`), `reduce_common` (→`bfloat16`).
- **Tier C (config coupling):** `bfloat16` — pulls `miopen/config.h` only for `MIOPEN_USE_RNE_BFLOAT16`.
- **Tier D (errors coupling):** `stringutils` (~61 sites, high churn), `tensor_layout` — both use only `MIOPEN_THROW(<string>)` (1 site in stringutils, 2 in tensor_layout).
- **Tier E (internally coupled, lives in `driver/`):** `driver/random.hpp` — couples to `miopen/env.hpp`, `miopen/kernel_tuning_mode.hpp`, `IsPerformanceLoggingEnabled()`.

Explicitly **out of scope:** heavy `miopen/errors.hpp` (~110 sites, bound to `miopenStatus_t` — stays in `src/`), `float8` impl (stays in `src/kernels/` per §1 decision 3), `tuple_utils` (doesn't exist on `develop`; net-new, not a move).

**Work items (stories):**
1. **Scaffold `common_utils` INTERFACE lib + clean leaves (Tier A+B)** — create `common_utils/CMakeLists.txt` (`add_library(miopen_common_utils INTERFACE)` + `MIOpen::common_utils` alias, `BUILD_INTERFACE` include dir only — not installed); `add_subdirectory(common_utils)` early from `projects/miopen/CMakeLists.txt` (before `src`); link (propagate includes) into `MIOpen`, `MIOpenDriver`, the test targets, `speedtests`, and `fin` (and `src/ck_impl` if it consumes a moved header). `git mv` the Tier A+B headers and rewrite include sites `<miopen/X.hpp>`→`<common_utils/X.hpp>` across `src driver test speedtests fin` (including intra-group includes inside the moved headers). No semantic risk. Model on #7253's `common_utils/CMakeLists.txt`.
2. **bfloat16 + reduce_common (Tier C)** — move `bfloat16`, dropping `#include <miopen/config.h>` in favor of an `INTERFACE` compile definition `MIOPEN_USE_RNE_BFLOAT16=0|1` on the target (keyed off the existing `option(...)`); add a `#ifndef MIOPEN_USE_RNE_BFLOAT16 → #error` guard so an undefined macro can't silently flip the rounding scheme (the references don't guard this). Then move `reduce_common` (depends on `bfloat16`, so it must follow). Build the float8/RTC host path explicitly to confirm `src/kernels/hip_f8_impl.hpp` still resolves `bfloat16` from `common_utils`.
3. **errors-lite + Tier D** — add `common_utils/errors.hpp`: a minimal `COMMON_THROW(msg)` that throws `std::runtime_error` (this is **not** a move of the heavy library `errors.hpp`, just a lightweight throw helper, matching #7291's `common_utils/errors.hpp`); move `stringutils` and `tensor_layout` onto it, replacing the 3 `MIOPEN_THROW` call sites. Gated by the Tier-D decision below — descope this story (ship Tier A–C only) if the C-API check is unverified.
4. **`random` (Tier E)** — create `common_utils/random.hpp` from `driver/random.hpp`, replacing `env::value(MIOPEN_DEBUG_DRIVER_PRNG_SEED)` with `std::getenv("MIOPEN_DEBUG_DRIVER_PRNG_SEED")` and dropping the `miopen/env.hpp`/`miopen/kernel_tuning_mode.hpp` includes; delete `driver/random.hpp`; repoint `test/random.hpp` and any other includers at `<common_utils/random.hpp>`. Carries a behavior change (drops the `IsPerformanceLoggingEnabled()` JSON-seed log line, leaving only the plain `PRNG seed: N`) — **isolate in its own commit**, call it out in the PR description, keep it revertible.
5. **CI layering guard (in this PR)** — add the guard framework + Phase-1 checks (`common_utils` has zero `miopen/` includes) and the initial whitelist with one-line reasons: `src/kernels/**` (RTC flat-include exception), `common_utils/errors.hpp` (intentional separate throw helper, not a violation), `src/kernels/hip_f8_impl.hpp` (commented-out `bfloat16` include), and `common_utils/float8.hpp` only if the optional forwarding shim is added. This is the guard framework the later phases extend.

**Decisions required (decision tickets):**
- **Tier D exception-type change** — moving `stringutils`/`tensor_layout` onto `COMMON_THROW` changes the thrown type from MIOpen's `miopenStatus_t`-bearing exception to `std::runtime_error`, and these headers run **inside** the library (stringutils ~61 sites), not just driver/test. Before implementing work item 3, confirm the C-API boundary catches `std::exception` (not only `miopen::Exception` inspecting `.status`) and maps it to a `miopenStatus_t`; inspect the TRY/CATCH macros in `src/include/miopen/errors.hpp` and their use at the `miopen.h` entry points. If ambiguous, descope Tier D rather than risk a behavior change.
- **`random` JSON-seed log line** — accept dropping the `{"prng_seed":N}` diagnostic (work item 4), or defer `random` until a public `IsPerformanceLoggingEnabled()` equivalent exists.
- **Forwarding shims** — follow §1 decision 4: redirect **all internal** include sites directly (no shims), *then* leave a forwarding shim at each old **public/installed** path that external consumers (MIFin, downstream tools) still build against. For Phase 1 this applies to the `common_utils` headers moved out of the installed `src/include/miopen/` namespace (e.g. `bfloat16.hpp`) — those old paths keep a shim; purely-internal moves do not. Do the internal rewrite first so the build breaks on any missed internal include, then add the external shims.

**Acceptance:** `grep -rn '#include' common_utils/include | grep miopen/` returns nothing and no `miopen/config.h` remains; Tier A–C moved with intra-group includes rewritten; `bfloat16` compile def is `INTERFACE` with the `#ifndef … #error` guard and unchanged rounding scheme; Tier D either moved (with the `std::exception` catch verified and documented) or explicitly descoped with a reason; `random` decoupled via `std::getenv`, `driver/random.hpp` deleted, `test/random.hpp` repointed, dropped JSON-seed line noted in the PR; all consumers (`src`, `driver`, `test`, `test/gtest`, `speedtests`, `fin`, `ck_impl` if applicable) build + link; smoke tests + `ctest` pass with no runtime/numeric change; CI guard rejects a cross-layer include and every intentional outlier is whitelisted with a reason.

### Phase 2 — Expand MIOpen public C API
**Depends on:** nothing structurally, but **should land before Phase 3/4** — `TensorDesc` (Phase 3) and the driver cleanup (Phase 4) consume these APIs. The tensor-query subset is the critical path for Phase 3.
**Reference:** §5 catalogue (distilled from #7252 / #7291), `./include/miopen/miopen.h` for current surface.

**Work items (stories):**
1. **API gap audit** — for each candidate in §5, determine: (a) does it already exist in `miopen.h`? (b) which internal symbol/header it replaces, and (c) how many driver/test sites need it. Produces the authoritative implement/skip list. *This unblocks sizing every other Phase-2 story.*
2. **Tensor query APIs** — `miopenGetTensorLayout`, `miopenGetTensorElementSpace`, `miopenIsTensorPacked`, `miopenGetTensorVectorLength`, `miopenGetTensorDescriptorV2` (those the audit confirms missing). Directly unblocks the 32 `miopen/tensor.hpp` driver includes via `TensorDesc`.
3. **Conv/pool padding-mode APIs** — `miopenGetConvolutionPaddingMode`, `miopenGetPoolingPaddingMode`.
4. **Solver-name APIs** — `miopenGetSolverName`, `miopenGetSolverIdByName`.
5. **Debug-flag APIs** — `miopenSetDebugFlag`/`miopenGetDebugFlag` + `miopenDebugFlag_t` enum (logging-quiet, find-enforce disable, warmup, naive-solver enable).
6. **CI layering guard (in this PR):** no new boundary here (Phase 2 only adds public API), but extend the guard to assert the new APIs are in the export set / public header, so later phases can rely on them.
- Each implementation story = declaration in `miopen.h` + implementation + export-list entry + a smoke test exercising the new entry point.

**Decisions required (decision tickets):**
- **Internal GEMM** (`gemm_v2.hpp`): add a public GEMM C API, or accept dropping the dependent feature (`gemm_driver`, per #7291)? *Confirmed: no `miopenGemm*` exists today.* Stakeholder sign-off needed — gates a Phase-4 work item.
- **`EnvEnableTF32()`**: expose as a debug flag, relocate, or drop.
- **Custom `bfloat16` type** at the public boundary: keep internal, or is the `common_utils` host type sufficient for consumers?

**Acceptance:** every internal symbol the driver currently needs has a public equivalent **or** an explicitly documented deferral/decision; each new API has a smoke test; library builds and exports cleanly; no runtime change to existing APIs.

### Phase 3 — Introduce `miopen_utils` (MIOpen Utilities) + `TensorDesc`
**Depends on:** Phase 1 (`common_utils`) + Phase 2 tensor-query APIs (the `TensorDesc` wrapper is built on them). Largest and highest-risk phase.
**Lib type:** **STATIC** — unlike `common_utils`, this layer has compiled sources (`cpu_*` refs), so it is the static-lib case from §1 decision 1. Wiring differs from Phase 1's INTERFACE lib (real archive, link order matters).

**Work items (stories):**
1. **Scaffold `miopen_utils` STATIC lib** — `miopen_utils/include/miopen_utils/*.hpp` + sources + `CMakeLists.txt` (model #7253/#7291); link into driver, tests, speedtests. Depends only on `common_utils` + `miopen/miopen.h`.
2. **Move-set inventory** — confirm the exact header/source list against the current tree before moving (today's `test/` candidates: `cpu_adam, cpu_bias, cpu_cat, cpu_conv, cpu_glu, cpu_groupnorm, cpu_kthvalue, cpu_layernorm, cpu_multimarginloss, cpu_prelu, cpu_reduce_util, cpu_rnn, cpu_softmarginloss, cpu_transformers_adam_w, fusionHost, gemm, network_data, rnn_util, seq_tensor_holder, serialize, tensor_holder, verify` — 22 files; re-measure, #7291 had more). Each that pulls a `miopen/` internal header is its own decouple-on-move task (mirror Phase 1's tiering).
3. **`TensorDesc` RAII wrapper** (from #7291) — wraps `miopenTensorDescriptor_t` using **only** the public C API; replaces `miopen::TensorDescriptor` value-type usage. Define the wrapper API surface from the operations driver/tests actually use. This unblocks the 32 `miopen/tensor.hpp` driver includes — expect missing APIs to surface here, feeding back to Phase 2.
4. **Eliminate `driver/ ↔ test/` cross-includes** — current: ~33 driver files include test-style verification headers, ~19 test files include driver headers. Both directions → 0 once the shared code lives in `miopen_utils`.
5. **CI layering guard (in this PR):** add `miopen_utils` checks — it may include only `common_utils/` + `miopen/miopen.h`; and assert 0 `driver/↔test/` cross-includes. Add any new outliers to the whitelist with a reason.

**Acceptance:** `grep` shows `miopen_utils/` includes only `common_utils/` and `miopen/miopen.h` (no other `miopen/` internal headers); 0 `driver/↔test/` cross-includes; driver/tests build tensors via `TensorDesc`; build + smoke + ctest pass, no runtime change.

### Phase 4 — Clean the driver
**Depends on:** Phase 2 (public APIs) + Phase 3 (`miopen_utils`, `TensorDesc`). After this, the driver pulls **zero** MIOpen internal headers.

**Decisions required (decision tickets):**
- **Driver-local vs. shared helpers** — #7252 used driver-local headers (`driver_errors`, `driver_log`, `driver_env`, `driver_tensor*`, `driver_ford`, `driver_utils`); #7291 kept a slimmer driver-local set (`driver_env`, `driver_log`) and pushed the rest into `miopen_utils`. Pick one before starting; it changes which Phase-4 stories exist. Plan currently leans toward the #7291 split.
- **`gemm_driver` fate** — confirm the Phase-2 GEMM decision landed; if no public GEMM API, removing `gemm_driver` (per #7291) needs stakeholder sign-off here.

**Work items (stories):**
1. **Replace `MIOPEN_THROW`** — 27 driver files; swap to `COMMON_THROW`/driver-local error per the helper decision above.
2. **Replace remaining internal includes** — after `tensor.hpp`/`errors.hpp` are handled via `TensorDesc`/error helper, clear the tail: `tensor_view_utils, logger, handle, tensor_extra, env, tensor_ops, rnn, kernel_tuning_mode, dropout, convolution, …` (see §3 inventory; re-measure). Group by header for reviewable commits.
3. **Extract `GPUMem`** — move out of `driver/driver.hpp` (currently lives there) into its own unit (fold in `extract_gpumem`, commit `7fa9d1c3fe8`).
4. **Resolve no-public-equivalent features** — execute the GEMM/TF32/bf16 decisions from Phase 2 (add API, relocate, or drop the dependent driver code).
5. **CI layering guard (in this PR):** add driver checks — `driver/` includes of `miopen/` resolve only to `miopen/miopen.h` (+ `miopen/config.h`); `grep 'MIOPEN_THROW' driver/` → empty.

**Acceptance:** driver depends only on `common_utils` + `miopen_utils` + `miopen/miopen.h` (+ `miopen/config.h`); zero internal headers; `grep 'MIOPEN_THROW' driver/` → nothing; smoke tests + ctest pass.

### Phase 5 — Clean the tests
**Depends on:** Phase 2 (public APIs) + Phase 3 (`miopen_utils`, `TensorDesc`). Lower risk than Phase 4 but broad.

**Work items (stories):**
1. **Classify test internal-access** — audit `test/` + `test/gtest/` for internal `miopen/` usage and bucket each site as (a) convertible to public API / util libs, or (b) deliberately testing a MIOpen internal. Produces the conversion worklist + the intentional-exception list.
2. **Convert bucket (a)** — switch those tests to public API + `common_utils`/`miopen_utils`. Group by area for review.
3. **Isolate & document bucket (b)** — keep deliberate internal access minimal, isolated, and commented so the CI guard can whitelist them with a reason.
4. **CI layering guard (in this PR):** finalize the test-layer checks and whitelist the deliberate internal-access points from bucket (b), each with a reason. After this phase the guard covers every layer boundary.

**Acceptance:** tests build against the layered structure; intentional internal-access points are explicit, minimal, and whitelisted; smoke + ctest pass.

---

### Cross-cutting work (not a phase PR)

**CI layering guard** is **not** a standalone epic — it lands **per-phase, inside each phase's PR**, kept with the changes it enforces (§1 decision 2). Each phase adds its own boundary checks + whitelist entries (see the "CI layering guard" work item in each phase). The guard framework is created in Phase 1 and extended through Phase 5, after which it covers every layer boundary.

**X — Symbol-leak baseline & verification** (§6) — a **task, not a PR** (no source change; just measurement runs).
- Establish a **baseline** (`nm`/`readelf` of the driver binary) capturing the internal MIOpen symbols it currently pulls — the "before" snapshot and the ultimate success measure beyond include hygiene.
- Re-run as a **comparison run after each phase**; the diff should shrink monotonically to zero internal symbols.
- **Acceptance:** post-refactor driver binary pulls no internal MIOpen symbols; a documented baseline + comparison method exists for future regression checks.

---

Track each phase with TaskCreate when execution begins.

---

## 5. New Public C API Catalogue (from references)

Collected from #7252 and #7291 — to be re-validated against current `miopen.h` before implementing (some may already exist):

- **Tensor queries:** `miopenGetTensorLayout`, `miopenGetTensorElementSpace`, `miopenIsTensorPacked`, `miopenGetTensorVectorLength`, `miopenGetTensorDescriptorV2`
- **Conv/pool:** `miopenGetConvolutionPaddingMode`, `miopenGetPoolingPaddingMode`
- **Solver names:** `miopenGetSolverName`, `miopenGetSolverIdByName`
- **Debug knobs:** `miopenSetDebugFlag` / `miopenGetDebugFlag` with a `miopenDebugFlag_t` enum (logging quiet, find-enforce disable, warmup mode, naive-solver enable)
- **Known gaps with no public equivalent yet:** internal GEMM (`gemm_v2.hpp`), `EnvEnableTF32()`, custom `bfloat16` type. Decide per-item: add public API, move to a util lib, or accept removal of the dependent feature (as #7291 did for `gemm_driver`).

---

## 6. Testing Strategy

- **Build gates per phase:** `MIOpen` library, `MIOpenDriver`, and the test suite must all build after every phase (each phase is "functionally equivalent, no runtime change").
- **Functional smoke tests** (from #7252 test plan):
  - `./build/bin/MIOpenDriver conv -n 1 -c 1 -H 32 -W 32 -k 1 -y 3 -x 3 -V 1`
  - `./build/bin/MIOpenDriver bnorm -n 1 -c 1 -H 32 -W 32 -V 1`
  - `./build/bin/MIOpenDriver pool -n 1 -c 1 -H 32 -W 32 -V 1`
- **`ctest`** regression run for unit tests.
- **Layering guard (grep-based, run in CI ideally):**
  - `grep -rn '#include.*miopen/' common_utils/` → empty
  - `grep -rn '#include' common_utils/ | grep miopen` → empty
  - driver/miopen_utils includes of `miopen/` resolve only to `miopen/miopen.h` and `miopen/config.h`
  - `grep -rn 'MIOPEN_THROW' driver/` → empty
  - 0 cross-includes between `driver/` and `test/`
- **Symbol-leak verification:** after linking, confirm the driver binary no longer pulls internal MIOpen symbols (e.g. `nm`/`readelf` diff against baseline) — this is the ultimate success measure beyond include hygiene.

---

## 7. Risks & Things to Watch

- **Scope/size.** The combined approach (#7291) ballooned to 244 files. Mitigate by shipping phase-by-phase; keep each PR functionally equivalent.
- **`miopen/tensor.hpp` (×32) is the hard part.** The driver uses `TensorDescriptor` as a value type everywhere. The `TensorDesc` wrapper + new tensor public API is the lever; expect this to be the largest single piece of work (Phase 3) and the most likely to surface missing API.
- **Kernel include-path / HIP-at-core question.** **Resolved by #7291** (HIPRTC flat-include constraint — see §1): RTC-included headers (`hip_float8.hpp`, `tensor_view.hpp`) stay in `src/kernels/` as documented exceptions. Carry the pattern forward; don't try to "fix" it inside this effort.
- **Config coupling.** `bfloat16.hpp` (and likely `float8`) depend on `miopen/config.h`; moving them requires replacing config macros with CMake defines without changing numeric behavior.
- **No-public-equivalent dependencies.** GEMM, TF32 enable, custom bf16. Each needs an explicit decision (add API / relocate / drop feature). #7291 dropped `gemm_driver` — verify that's acceptable to stakeholders.
- **Forwarding-shim policy (§1 decision 4).** Shims are **kept** at old public/installed paths for external consumers (don't break MIFin/downstream), but **all internal** usage must be redirected to the new location. Implement the internal redirect first *without* shims (so the build fails on any missed internal include), verify, then add the external shims. Internal code including a shim path is a layering violation the CI guard should catch — the shim exists only for out-of-repo consumers.
- **Build-system breadth.** `CMakeLists.txt` changes span `projects/miopen`, `src/`, `src/ck_impl/`, `driver/`, `test/`, `test/gtest/`, `speedtests/`. Easy to miss one and break a downstream target.
- **Stale references.** The reference branches are from May 2026 and `develop` has moved on; treat their file lists/symbol counts as a guide, not gospel — re-measure against current tree before each phase.
- **Don't import the cruft.** #7291 carried stray `.claude/` plan/lock files and worktree artifacts. When mining it, copy only intended source — never the tree wholesale.
- **`fix_symbol_leaks` vs `fix_symbol_leaks_2`.** Two versions exist; reconcile which represents the author's better thinking before following either.

---

## 8. Summary

The objective is a clean four-layer MIOpen: **Core Utilities (`common_utils`) → MIOpen library (public C API) → MIOpen Utilities (`miopen_utils`) → Driver & Tests**, with no internal-symbol leakage across boundaries. Today the driver and tests are deeply tangled with MIOpen internals (53 driver files include internal headers, ~50 cross-includes between driver and tests, 27 `MIOPEN_THROW` users in the driver).

Prior, unmerged work by brpepers is the reference. It's one linear evolution culminating in **#7291 (`tensor_desc_wrapper_rebased`)**, the complete superset and primary thing to read; **#7253** is the foundational explainer and **#7252** an alternative design kept only for contrast (see §2).

The recommended path is to **start fresh on this branch** and rebuild the layering **incrementally** — (1) `common_utils`, (2) expand the public C API, (3) `miopen_utils` + the `TensorDesc` wrapper, (4) driver cleanup, (5) test cleanup — with each phase staying functionally equivalent and gated by build + smoke + `ctest` + grep-based layering checks. The **HIP-at-core / kernel-include-path** question is already resolved (HIPRTC flat-include constraint → RTC headers stay in `src/kernels/` as documented exceptions, per #7291); still decide the **header-only vs. static-lib** choice up front, and make explicit per-item decisions for the handful of internal dependencies (GEMM, TF32, custom bf16) that have no public equivalent today.
