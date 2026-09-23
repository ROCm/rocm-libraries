# Matching-Infinities Validator: a TOML knob for the bundle harness

- **Ticket**: ALMIOPEN-2610
- **Branch**: `users/ggilbert/ALMIOPEN-2610/rocke-gfx950-dense-ingestor` (PR #12311)
- **Worktree**: `C:/develop/claude_workspace/worktrees/rocm-libraries/ALMIOPEN-2610`
- **Plan drafted**: 2026-09-23

---

## 1. Request Summary

Add a "matching infinities" comparison policy to the bundle-driven integration-test harness
in `dnn-providers/integration-tests/`, so an engine's TOML config can select, per tensor, a
validator that treats same-signed infinities in the reference and the device output as equal
rather than as a failure.

This closes a gap identified in review of merged PR #11508. The test SDK gained
`MatchingInfinities::ACCEPTED`, but only hand-written C++ graph tests that build their own
validator can opt into it. The shared bundle harness hardcodes `REJECTED` through
`createAllCloseValidator` and exposes no knob at all. Reviewer adickin-amd, on
`CpuFpReferenceValidation.hpp:26`: *"This looks like a feature that the integration tests are
going to need to have available in the testing bundles."* No backlog ticket exists (Jira was
searched; nothing found).

Filing a follow-up ticket and leaving this branch alone **was** a viable option, and is what
this ticket's AC6 would normally direct. It was offered and declined; the user chose to land
the harness plumbing here. See **§2.5**, which states that tension and the accepted cost
plainly — this plan is not the only thing that could have happened.

---

## 2. Fixed Point

### 2.1 Target outcome

An engine TOML config can write:

```toml
[[validator_overrides]]
filters   = ["*SomeTest*"]
tensors   = ["*::LSE"]
validator = "allclose_matching_infinities"
```

and the named output tensors of the matching tests are graded by
`hipdnn_test_sdk::utilities::createAllCloseMatchingInfinitiesValidator`, at the atol/rtol the
harness resolved, instead of by the default allclose validator.

### 2.2 Hard constraints (design CLOSED — do not reopen)

**Option A** is the signed-off approach:

1. New enumerator `ALLCLOSE_MATCHING_INFINITIES` on `bundle::ValidatorKind`
   (`src/harness/bundle/OutputComparison.hpp:52-56`) **and** its mirror
   `ValidatorOverrideKind::ALLCLOSE_MATCHING_INFINITIES`
   (`src/harness/TestSettings.hpp:44-48`).
2. New static factory `ComparisonTolerance::allCloseMatchingInfinities(float atol, float rtol)`.
3. TOML spelling: `validator = "allclose_matching_infinities"`.
4. Selected **only** by a `[[validator_overrides]]` entry, exactly as `rms` is. Nothing
   selects it by default, on any path.
5. `ValidatorKind::RMS` is the worked precedent. Mirror its structure, its guard style, and
   its test coverage.
6. The new kind carries **real atol/rtol** (unlike `rms`, which zeroes them).
7. **No change to `projects/hipdnn/test_sdk/`.** The SDK API already exists and is used
   as-is.

> **Reading note on constraint 5.** "Mirror its test coverage" is scoped to the *unit* tests
> §7.2 enumerates. It does **not** mean the new kind will end up as well covered as RMS: RMS
> additionally has live integration coverage from three committed TOML entries, which the new
> kind is forbidden to have (§2.3). See R1 — that asymmetry is real, is not closed by this
> plan, and must not be papered over in the PR.

### 2.3 Non-goals (do NOT do any of these)

- Migrating the ASM SDPA STATS test off hip-kernel-provider's local harness onto the shared
  TOML mechanism. Out of scope; the user has explicitly reserved that scope decision.
- Any change to `dnn-providers/hip-kernel-provider/src/integration_tests/IntegrationGraphVerificationHarness.hpp`
  or `.../asm_sdpa_engine/IntegrationGpuSdpaForward.cpp`.
- Lifting the stats decline in `Gfx950AttentionDenseNative.cpp:519-525`.
- Adding a stats tensor to the gfx950 descriptors, or adding any new test bundle.
- Any change to `src/harness/bundle/BundleReferenceValidationHarness.cpp:119-150`. That path
  grades golden data with allclose at the default tolerance and never selects a non-default
  validator. This is **deliberate policy, not a gap**: the code comment at `:129-131` states
  it ("a TOML override belongs to an engine and must not loosen the gate on our own data. For
  the same reason this path never selects a non-default validator — allclose always"), and
  `README.md:457-461` documents it for users. Do not change it and do not file it.
- Any change to the bundle JSON schema or `IntegrationTestBundle.hpp`.
- Any change to `CpuFpReferenceValidation.hpp`, `CpuFpReferenceMiopenRmsValidation.hpp`, or
  anything else under `projects/hipdnn/test_sdk/`.
- Any GPU-dependent **unit** test. The oracle in §7.1 runs on CPU. (The one-off join check in
  §7.5 does use the local GPU; it is a manual verification step, not a test that ships.)

**`[[validator_overrides]]` entries in engine TOML — two separate rules, previously
conflated:**

- **Shipping prohibition — stands, absolutely.** No `[[validator_overrides]]` entry naming
  `allclose_matching_infinities` may be **committed** to `HIP_MLOPS_ENGINE.toml`,
  `ASM_SDPA_ENGINE.toml`, `hipkernel_Gfx950AttentionDense.toml`, or any other engine config.
  §3.5 proves any such entry would be dead config.
- **Verification permission — newly granted (user decision, 2026-09-23).** A **temporary,
  never-committed** config selecting the new validator IS sanctioned, solely for the one-off
  join check that proves TOML text actually reaches the new validator. §7.5 specifies it, and
  it uses a **scratch file outside the repository** so no tracked file is edited at all.

### 2.4 The deliverable has zero live consumers, on purpose

This ships **knob + unit tests + schema documentation, with nothing selecting it**. That is
the intended end state of this ticket, not an oversight. Evidence is in §3.5. The **PR body
must state this plainly** so a reviewer does not go looking for the consumer (owned in §6.3).

### 2.5 This work maps to none of ALMIOPEN-2610's acceptance criteria — disclosed, accepted

State this in the PR body and in the ticket. It is not a footnote.

- **This plan satisfies none of the six acceptance criteria on ALMIOPEN-2610.** It is
  adjacent work landing on the same branch.
- **AC6 reads**: *"Any defect or gap found in the integration skill or tooling during the run
  is filed against ALMIOPEN-2446."* A harness/tooling gap of exactly this kind is what AC6
  routes to ALMIOPEN-2446 **as a filing, not as a fix**. Landing the fix here is in direct
  tension with that criterion.
- **The user was offered three options on 2026-09-23** and chose (b):
  - (a) file a follow-up against ALMIOPEN-2446 and leave this branch alone — the AC6-conformant path;
  - (b) **chosen** — add the harness plumbing here now;
  - (c) additionally add gfx950 stats coverage so the knob has a real consumer.
- **The user made this choice after being told the cost**, which is:
  1. The code ships with **no consumer exercising it end to end** in CI (§3.5, §2.4).
  2. It **grows this PR's blast radius** out of the gfx950 dense-ingestor work and into shared
     integration-test harness code that every provider's suite runs through
     (`OutputComparison`, `TestSettings`, `TomlGuards`).
  3. A reviewer arriving via the ticket will find a diff that no acceptance criterion asked
     for, and must be told why up front rather than discovering it.
- **Accepted, with the mitigation in §7.5**: the one-off join check is what buys back part of
  cost (1) — it demonstrates the mechanism works end to end at least once, by hand, even
  though nothing in CI will keep doing so.

---

## 3. Research Findings

### 3.1 Test SDK API — already exists, consumed as-is

`projects/hipdnn/test_sdk/include/hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp`:

| Line | Fact |
|------|------|
| `:24-28` | `enum class MatchingInfinities { REJECTED, ACCEPTED };` |
| `:246-253` | `createAllCloseValidator(DataType, float atol = eps, float rtol = eps)` hardcodes `REJECTED` at `:252`. No overload takes the policy. |
| `:261-269` | `createAllCloseMatchingInfinitiesValidator(DataType, float atol, float rtol)` — atol/rtol **not** defaulted. **This is the function to call.** |
| `:200-207` | `validation_detail::rejectMatchingInfinitiesRequest` throws `std::runtime_error("Matching infinity allClose validator requires a floating point data type")` for INT8/UINT8/INT32 (`:230-238`). |
| `:240` | Any other `DataType` throws `std::runtime_error("Unsupported data type for allClose validator")`. |
| `:76-94` | Semantics: **only same-signed** matching infinities short-circuit as equal (`:79-83` returns the running result untouched). NaN, opposite-signed inf, and finite-vs-inf all still fail at `:85`. Finite elements compare by atol/rtol as usual at `:96-111`. |

FLOAT / HALF / BFLOAT16 / DOUBLE accept the ACCEPTED policy; the three integer types throw.

### 3.2 Seams to change

All under `dnn-providers/integration-tests/`. Line numbers verified 2026-09-23 against the
worktree.

| File:line | Current shape | What it needs |
|-----------|---------------|---------------|
| `src/harness/bundle/OutputComparison.hpp:41-56` | Doc comment + `enum class ValidatorKind { ALLCLOSE, RMS };` | Third enumerator, appended; doc comment extended |
| `src/harness/bundle/OutputComparison.hpp:58-77` | `struct ComparisonTolerance { float atol; float rtol; ValidatorKind kind; float rmsThreshold; }` with `allClose(atol,rtol)` and `rms(threshold)`. Both factories use aggregate init and compile unchanged when an enumerator is added. | One new factory only |
| `src/harness/bundle/OutputComparison.cpp:33-76` | `makeValidator` — `switch(tolerance.kind)`. `ALLCLOSE` at `:40`. `RMS` at `:44` wrapped in `try/catch(const std::exception&)` returning a NOT-APPLICABLE report (`:49-68`). `default: throw std::invalid_argument` at `:70-74`. | New case, **inside its own try/catch of the same shape** — it is glob-selected like RMS and throws on integer dtypes |
| `src/harness/bundle/OutputComparison.cpp:78-119` | `formatMismatchReport`; `useRms = kind == RMS` at `:87` | **No change** — see Q3 |
| `src/harness/TestSettings.hpp:44-55` | `enum class ValidatorOverrideKind { ALLCLOSE, RMS };` ("Mirrors bundle::ValidatorKind, kept separate so TestSettings stays free of the comparison headers") + `struct ValidatorOverride { kind; rmsThreshold; }` | Third enumerator, appended |
| `src/harness/TestSettings.hpp:62-105` | Documented TOML schema in a header comment block; validator rules at `:91-98` | Extend |
| `src/harness/TestSettings.hpp:372-441` | `parseValidatorOverride`. Strict: `allclose`-with-`rms_threshold` rejected at `:403-414` with the standard stated inline ("an entry that does not say exactly what it means is a load error"); unknown names rejected at `:419-423`. | Accept the third name; **reuse** the `:409-414` "must not carry rms_threshold" guard; update the `:422` expected-names message |
| `src/harness/TomlGuards.hpp:58-71` | `findTomlRmsThreshold(testName, tensorLabel) -> std::optional<float>` | **The one non-mechanical edit.** Returns a bare float; structurally cannot carry a non-RMS kind. Must be generalised to return the `ValidatorOverride`. |
| `src/harness/TomlGuards.hpp:82-102` | `gradingForTensor` — the single site translating a TOML override into a `ComparisonTolerance` | Third branch |
| `README.md:397-464` | Prose schema doc for `[[validator_overrides]]`, incl. the `"allclose"`/`"rms"` list at `:438` and the RMS dtype caveat at `:441-443` | Extend |
| `dnn-providers/hip-kernel-provider/config/HIP_MLOPS_ENGINE.toml:25-33` | Canonical commented schema block naming `validator: "allclose" or "rms"` | Extend the **comment only**; add no entry (§3.5) |

Unchanged consumers that pick the new kind up for free:
`src/harness/bundle/IntegrationBundleVerificationHarness.cpp:626-638` (resolves every bundle
output through `gradingForTensor` at `:637`) and
`src/harness/IntegrationGraphVerificationHarness.hpp:287` (the same for C++ graph tests).

### 3.3 Tolerance provenance

`src/harness/tolerance/ToleranceResolver.hpp` — `toleranceForNodeAttributes<T>` (`:74-76`
routes `SdpaAttributes` to `tol::sdpa::getToleranceFwd<T>()`), `maxAcrossNodes` (`:118`),
`defaultTolerance` (`:166`). `IntegrationBundleVerificationHarness.cpp:636` calls
`defaultTolerance(...)` and passes the result as **both** atol and rtol into
`gradingForTensor`. The new kind reuses that value untouched.

### 3.4 Existing tests to mirror

`tests/TestOutputComparison.cpp` — RMS precedent, to copy test-for-test:
`RmsAcceptsANearZeroElementThatAllcloseRejects` (`:291`),
`RmsStillRejectsDriftLargeAgainstTheTensorScale` (`:316`),
`RmsFailureReportsItsThresholdNotAtolRtol` (`:333`), `ValidatorKindIsChosenPerTensor`
(`:354`), `RmsOnAnUnsupportedDataTypeIsReportedNotThrown` (`:413`),
`AllcloseStillGradesIntegerOutputs` (`:442`), `UnhandledValidatorKindIsRefused` (`:461`).

Fixtures available in that file's anonymous namespace (`:32-115`): `K_UID_A` = 5 (named
`"y_out"`, FLOAT), `K_UID_B` = 4 (unnamed → label `"uid=4"`, FLOAT `[1,3,1,1]`), `K_UID_INT` =
6 (named `"counts"`, INT32); helpers `makeGraphBuffer()`, `floatTensor`, `floatTensor3(attrs,
v0, v1, v2)` (writes three arbitrary floats — accepts infinities and NaN), `intTensor`,
`exact()`.

`tests/TestTestSettings.cpp` — parser precedent: `MatchesOnBothNameAndTensorGlob` (`:336`),
`LaterEntriesTakePrecedence` (`:377`), `ThrowsOnUnknownValidatorKind` (`:423`),
`ThrowsOnRmsWithoutThreshold` (`:439`), `ThrowsOnAllcloseWithRmsThreshold` (`:473`). Uses a
`TempTomlFile` RAII helper and constructs `TestSettings` directly — **no singleton**.

`tests/TestTomlGuards.cpp` — calls `findTomlRmsThreshold` by name at `:11`, `:52`, `:81`, and
`gradingForTensor` at `:12`, `:90`. **Renaming the helper breaks this file**; it is in no
other stream's ownership list and is named explicitly in Task 4 below.

### 3.5 Why nothing will consume this knob

Do not invent a consumer. Verified:

- This branch's 7 SDPA bundles
  (`integration-test-bundles/{quick,standard}/SdpaFwdRuntimeScale/bshd/**`) all declare
  `"stats_tensor_uid": null` and `"generate_stats": null` (lines 32 and 40 of each). There is
  no LSE tensor to grade. Re-verified 2026-09-23: exactly 7 JSON files, none lacking the null
  declaration.

  > **Read the layout token carefully — `bshd`, not `bhsd`.** They differ by one transposed
  > letter and mean different things here. `SdpaFwdRuntimeScale/**bhsd**/{bf16,fp16}/hd128_nomask_batch_stats/SmallStats`
  > **does** exist and **does** carry an `LSE` tensor — but those are pre-existing bundles
  > (added by #9600), not this branch's, and they are `nomask`, so they produce no `-inf`
  > either. A reader scanning `SdpaFwdRuntimeScale` will hit them and must not read them as a
  > counterexample to the sentence above.
- The gfx950 pack hard-declines any stats-bearing graph:
  `dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/packs/Gfx950AttentionDenseNative.cpp:519-525`
  returns `std::nullopt`.
- The one place infinity bites today is
  `dnn-providers/hip-kernel-provider/src/integration_tests/asm_sdpa_engine/IntegrationGpuSdpaForward.cpp:74-88`.
  That test uses hip-kernel-provider's **own separate** harness
  (`.../src/integration_tests/IntegrationGraphVerificationHarness.hpp:225-240`), which never
  uses `ComparisonTolerance` and has a `registerValidator` overload taking a caller-built
  validator. It does not read engine TOML for this, so a TOML entry cannot reach it.
- Bundles on develop that *do* name a stats tensor call it `"LSE"` (e.g.
  `integration-test-bundles/quick/SdpaFwd/bhsd/bf16/hd128_nomask_batch_stats/SmallStats/SmallStats.json:131`,
  and the two `SdpaFwdRuntimeScale/bhsd/{bf16,fp16}/hd128_nomask_batch_stats/SmallStats`
  bundles noted above), but **all** are `nomask`, so none produces `-inf`. Evidence that
  label-based selection would work — not a consumer.

An entry in `hipkernel_Gfx950AttentionDense.toml` or `ASM_SDPA_ENGINE.toml` would be dead
config. Add none.

### 3.6 Build and test facts confirmed in this worktree

- Unit-test target: **`hipdnn_integration_tests_unit_tests`**
  (`dnn-providers/integration-tests/tests/CMakeLists.txt:9-92`). `OutputComparison.cpp` is
  compiled into it at `:14`; `TestTestSettings.cpp` `:30`, `TestTomlGuards.cpp` `:31`,
  `TestOutputComparison.cpp` `:67`.
- The target is registered through `add_unit_test_target` →
  `dnn-providers/cmake/Tests.cmake:398`. Because `DNN_PROVIDER_TEST_CATEGORY_YAMLS` is set
  (`Tests.cmake:359-362`), the raw target is **not** registered as a bare CTest test; tiered
  suites are generated instead. Confirmed by `ctest -N`: the CTest name is
  **`hipdnn_integration_tests_unit_tests_quick_suite`**.
- **There is no `hipdnn-integration-tests-unit-check` ninja target in this configuration.**
  `references/hipdnn.md` lists one; `ninja -t targets` shows only
  `hipdnn-integration-tests-{,quick-,standard-,ffm-quick-}check`. The tier targets also drag
  in `hipdnn_gpu_ref_tests_*` (GPU) and the golden-data tests, so none of them is the oracle.
- Warning flags: `-Werror -Wall -Wextra ... -Wswitch-default`
  (`dnn-providers/integration-tests/CMakeLists.txt:77-98`). **Every `switch` must carry a
  `default:`** — exhaustiveness-by-omission is not available, which is why `makeValidator`
  has `default: throw`.
- Existing build dir `build/` is configured with
  `ROCM_LIBS_ENABLE_COMPONENTS=hipdnn;hipdnn-integration-tests;hip-kernel-provider`,
  `CMAKE_BUILD_TYPE=Release`, `HIPDNN_ENABLE_KERNEL_INGESTOR=ON`, `ENABLE_CLANG_TIDY=OFF`.
  `ninja -C build hipdnn_integration_tests_unit_tests` runs clean from a plain Git Bash shell
  (no VsDevCmd needed — the toolchain is pinned in the cache). Verified 2026-09-23.
- Baseline oracle run: `ctest -R hipdnn_integration_tests_unit_tests_quick_suite` **passes in
  6.17 s** from `build/dnn-providers/integration-tests`. Verified 2026-09-23.
- **GOTCHA, verified:** running the binary with a narrow `--gtest_filter` that excludes
  `TestTestConfig.cpp` makes four `TestTomlGuards` tests **fail**
  (`CheckTomlSkipReturnsNulloptWhenNoSettings`,
  `ApplyTomlToleranceOverrideReturnsFalseWhenNoSettings`,
  `FindTomlRmsThresholdReturnsNulloptWhenNoSettings`,
  `GradingForTensorKeepsTheCallersToleranceWhenNoSettings`). They depend on
  `TestConfigInitialized::SetUpTestSuite` having initialized the `TestConfig` singleton
  earlier in the same binary — stated in the comment at `TestTomlGuards.cpp:56-61`. **The
  oracle must run the whole binary / the whole quick suite, never a narrow filter.**
- `TestConfig` is a one-shot singleton: `TestConfigInitialized.DoubleInitializeThrows`
  (`TestTestConfig.cpp:222-226`). It is initialized in this binary **without** a settings
  file, so no unit test in this binary can put a TOML in front of `gradingForTensor`. See
  Risk R1.

---

## 4. Rejected Alternatives

Each rejection is scoped to the condition that produced it. If the condition lifts, the
rejection lapses.

### B — orthogonal `MatchingInfinities` field on `ComparisonTolerance`

Rejected on three counts, **while** the parser keeps its current strictness standard and the
RMS validator keeps its current inf/NaN handling:

1. `validator = "rms"` + `matching_infinities = true` would be representable but meaningless.
   Refusing it needs a new parser guard written to the standard already set at
   `TestSettings.hpp:403-414` — *"an entry that does not say exactly what it means is a load
   error"*. Option A makes the combination **unrepresentable** and needs no guard at all.
2. B's headline benefit — a pre-wired TOML key for the day RMS wants infinity-acceptance — is
   illusory. `CpuFpReferenceMiopenRmsValidation.hpp:68-77` currently hard-rejects any inf or
   NaN, and accepting matched infinities in an aggregate metric requires excluding them from
   `squareDifference`, `elementCount`, **and** `maxRefMagnitude`/`maxImplMagnitude`
   (`:53-55`, `:121-143`). That is a new validator implementation, which would arrive with
   its own name regardless — so the pre-wired key buys nothing.
3. `MatchingInfinities` is a **per-element verdict override** by construction:
   `CpuFpReferenceValidation.hpp:79-83` returns the running result untouched. Modelling it as
   kind-orthogonal asserts a generality the implementation does not have.

### C — defaulted `MatchingInfinities` parameter on `createAllCloseValidator`

Rejected **while** `createAllCloseValidator` remains a public test-SDK entry point with
roughly 50 call sites. A defaulted parameter reopens the explicit-named-opt-in design that
#11508 deliberately settled on, and it changes a public signature under
`projects/hipdnn/test_sdk/` — out of bounds per §2.3.

### D — carry the policy in bundle data (sidecar metadata)

Rejected **while** TOML already provides an equivalent per-tensor knob and the graph JSON is
schema-converted verbatim. D duplicates the knob, and the graph JSON structurally cannot hold
it. Its one unique property — reaching `BundleReferenceValidationHarness`, which reads no
engine TOML — is an explicit non-goal here (§2.3).

---

## 5. Questions & Answers

**Q1. Does a `[[tolerance_overrides]]` entry still apply when a `[[validator_overrides]]`
entry selects `allclose_matching_infinities`? — ANSWERED: yes.**
Derived, not chosen. Today an explicit `validator = "allclose"` override falls through
`TomlGuards.hpp:87-101` to `applyTomlToleranceOverride` and then `allClose(atol, rtol)` — the
tolerance override applies. The closed design states the new kind "must carry real atol/rtol,
unlike `rms` which zeroes them". The new kind is allclose plus a per-element infinity policy;
its tolerance semantics are identical. Applying the same resolution path is the mirror of
existing allclose behaviour, and *refusing* it would be the novel behaviour.

**No unit test pins this.** The behaviour lives entirely in the
`applyTomlToleranceOverride(testName, atol, rtol)` call inside Task 4 step 3's
matching-infinities branch — which is precisely the glue R1 concedes is uncoverable in the
unit binary. T14 is a bare `ComparisonTolerance` factory round-trip: it never calls
`gradingForTensor` and never calls `applyTomlToleranceOverride`, so it cannot answer Q1 and
must not be credited with doing so. **Q1 is discharged by the §7.5 join check instead**, which
is deliberately designed to carry a `[[tolerance_overrides]]` entry alongside the validator
entry so the answer is read directly off the logged atol/rtol. Verified observable — see
§7.5's control probe.

**Q2. Is a `[[validator_overrides]]` entry with `validator = "allclose_matching_infinities"`
and an `rms_threshold` a load error? — ANSWERED: yes.**
It reuses the existing `TestSettings.hpp:409-414` guard verbatim, per the closed design
("must reuse that 'must not carry rms_threshold' guard"). Pinned by test T11.

**Q3. Does `formatMismatchReport` change for the new kind? — ANSWERED: no.**
`useRms` at `OutputComparison.cpp:87` is `kind == ValidatorKind::RMS`, so the new kind takes
the atol/rtol path unchanged. That is correct rather than merely convenient: under the new
kind, atol and rtol **are** what graded every finite element, and an infinity disagreement is
reported by the per-element diff that already runs. RMS needed the report changed because its
threshold, not atol/rtol, decided the verdict; the new kind has no such divergence. Pinned by
test T8.

**Q4. What is the new name for `findTomlRmsThreshold`? — SETTLED: `findTomlValidatorOverride`,
returning `std::optional<ValidatorOverride>`.**
The closed design already mandates the generalisation ("It must be generalised to return the
`ValidatorOverride` (or equivalent)"); only the identifier was left to pick, and this one
matches the `TestSettings::findValidatorOverride` it wraps.

**Q5. Does `gradingForTensor` gain a guard for an unhandled `ValidatorOverrideKind`? —
SETTLED: yes, `default: throw std::invalid_argument`, mirroring `makeValidator:70-74`.**
The closed design says to mirror RMS's guard style. `-Wswitch-default` (§3.6) makes a
`default:` mandatory on any switch, so the choice is only between `default: throw` and
`default:` falling through to allclose — and silently grading an unknown kind as allclose is
exactly the failure `makeValidator`'s guard exists to prevent. It is unreachable through the
parser (which accepts three spellings and produces three enumerators) and therefore carries
no unit test, exactly as the analogous throw does not on the `ValidatorOverrideKind` side
today. Recorded as Risk R2.

---

## 6. Implementation Plan

### 6.1 Streams: one. Justification.

**This is a single sequential stream. Do not split it.**

Every edit lands in one subsystem — `dnn-providers/integration-tests/src/harness/` plus its
three test files — and the edits are strictly ordered by data dependency:

- The enumerators (Task 1) must exist before the parser can produce one or a test can name
  one.
- The `findTomlRmsThreshold` → `findTomlValidatorOverride` generalisation (Task 4) is the
  gate the brief calls out: nothing can express a third kind through `TomlGuards` until it
  lands, and it simultaneously edits `TomlGuards.hpp` and `TestTomlGuards.cpp`.
- `OutputComparison.{hpp,cpp}` is touched by Tasks 1 and 3; `TestSettings.hpp` by Tasks 1 and
  2 (Task 5 owns only `README.md` and `HIP_MLOPS_ENGINE.toml`). Two parallel streams would
  collide on both files immediately.

Total diff is roughly 40 source lines plus ~180 test lines. Parallelism here would cost more
in merge coordination than it saves. **No new worktrees** — all work happens in the existing
`ALMIOPEN-2610` worktree (rocm-libraries budget: 3, used: 0).

### 6.2 Sub-task order (one Implementor, sequential)

Each numbered task ends with a run of the local oracle (§7.1). Tasks 2–5 are planned
**test-first**: the test is written and observed **failing as an assertion** (not as a compile
error) before the production change, then observed passing. Each task flips at least one
distinct branch from red to green, per the several-branches-one-path rule.

---

#### Task 1 — Foundation: both enumerators plus a stub factory (no behaviour change)

*Blocks: Tasks 2, 3, 4. Blocked by: nothing.*

Files owned: `src/harness/bundle/OutputComparison.hpp`, `src/harness/TestSettings.hpp`.

1. Append `ALLCLOSE_MATCHING_INFINITIES` to `bundle::ValidatorKind`
   (`OutputComparison.hpp:52-56`), **after** `RMS`. Extend the doc comment above it (`:41-51`)
   in the same register as the RMS paragraph: say what the kind does (same-signed infinities
   in reference and device output compare equal; NaN, opposite-signed infinities and
   finite-vs-infinite still fail; finite elements graded by atol/rtol as usual), why it exists
   (an output whose correct value is infinite on both sides, such as an SDPA forward
   log-sum-exp row that is fully masked), and that nothing selects it on its own — only an
   engine's TOML `[[validator_overrides]]` entry naming the tensor.
2. Append `ALLCLOSE_MATCHING_INFINITIES` to `ValidatorOverrideKind`
   (`TestSettings.hpp:44-48`), after `RMS`.
3. Add the factory to `ComparisonTolerance` (`OutputComparison.hpp:68-76`):

   ```cpp
   static ComparisonTolerance allCloseMatchingInfinities(float atolIn, float rtolIn)
   {
       // STUB for Task 1 only — Task 3 replaces ALLCLOSE with
       // ValidatorKind::ALLCLOSE_MATCHING_INFINITIES.
       return ComparisonTolerance{atolIn, rtolIn, ValidatorKind::ALLCLOSE, 0.0f};
   }
   ```

   The stub exists so that the Task 3 tests can be written against a symbol that **links**,
   and therefore fail as assertions rather than as compile errors.

   > **The stub must never be committed, and must never be reported as done.** A factory named
   > `allCloseMatchingInfinities` that returns `ValidatorKind::ALLCLOSE` is *precisely* the
   > silent miscompare that `OutputComparison.cpp:70-74`'s `default: throw` exists to prevent:
   > it grades a tensor with a validator nobody chose, and every test passes. The technique is
   > sanctioned only as a transient scaffold inside this task. Do not stop work, hand off, or
   > report between Task 1 and the completion of Task 3a — if work must be set aside there,
   > remove the stub factory entirely first.

**Acceptance**: the suite is still green and nothing behaves differently. There is no red for
this task by construction (§7.3).

---

#### Task 2 — TOML parser accepts the third name

*Blocked by: Task 1. Blocks: Task 5 (docs consistency).*

Files owned: `src/harness/TestSettings.hpp`, `tests/TestTestSettings.cpp`.

1. **Red first.** Add T9, T10, T11 (§7.2) to `TestTestSettings.cpp`. Run the oracle; T9 and
   T10 must fail (the `TestSettings` constructor throws "unknown validator") and T11 must fail
   (nothing throws yet where it should — actually it throws for the *wrong* reason, so assert
   the kind is parsed in T9 first and only then expect T11's throw; if T11 is green for the
   wrong reason, note it in the checklist and rely on T9/T10's red).
2. Restructure `parseValidatorOverride` (`TestSettings.hpp:372-441`) so the
   "must not carry `rms_threshold`" check at `:409-414` is **shared** between `"allclose"` and
   `"allclose_matching_infinities"` rather than copied. Keep the existing comment's reasoning
   attached to the shared check.

   > **Order matters, and getting it wrong stays green.** The unknown-name check **must run
   > first**, before the shared threshold guard. If the guard is hoisted above it, then
   > `validator = "relative-rms"` **+** `rms_threshold = 1e-4` — which is exactly the input
   > `ThrowsOnUnknownValidatorKind` uses (`TestTestSettings.cpp:423-437`) — stops reporting
   > *"unknown validator"* and starts reporting *"also sets rms_threshold"*, sending an
   > operator to fix the wrong line. The suite would not notice: that test asserts only
   > `EXPECT_THROW(..., std::runtime_error)`. Required shape:

   ```cpp
   const bool isAllClose      = (*validator == "allclose");
   const bool isMatchingInf   = (*validator == "allclose_matching_infinities");
   const bool isRms           = (*validator == "rms");

   if(!isAllClose && !isMatchingInf && !isRms)   // FIRST — name before content
   {
       throw std::runtime_error(std::string(K_SECTION) + " entry has unknown validator '"
           + *validator
           + R"~(' (expected "allclose", "allclose_matching_infinities" or "rms"))~");
   }

   if(!isRms)   // the shared guard, reached only for a name we recognise
   {
       if((*table)["rms_threshold"]) { throw ... /* existing message + reasoning */ }
       parsed.kind = isAllClose ? ValidatorOverrideKind::ALLCLOSE
                                : ValidatorOverrideKind::ALLCLOSE_MATCHING_INFINITIES;
       parsed.rmsThreshold = 0.0f;
       return parsed;
   }
   // rms path below, unchanged
   ```

3. **Strengthen `ThrowsOnUnknownValidatorKind` in place** (`TestTestSettings.cpp:423`) so the
   ordering above is actually pinned: replace the bare `EXPECT_THROW` with a `try`/`catch`
   that asserts `what()` contains `unknown validator`. Without this assertion the hoisting
   hazard is invisible to the suite. This is the one existing test this plan modifies rather
   than merely preserving.
4. The unknown-name message now lists all three spellings (shown in step 2's snippet).
5. Extend the header schema comment block (`TestSettings.hpp:62-105`) — both the example at
   `:71-75` and the prose rules at `:91-98`.

**Acceptance**: T9, T10, T11 green; `ThrowsOnUnknownValidatorKind` (`:423`),
`ThrowsOnRmsWithoutThreshold` (`:439`), `ThrowsOnAllcloseWithRmsThreshold` (`:473`) and
`LaterEntriesTakePrecedence` (`:377`) all still green.

---

#### Task 3 — The comparison actually accepts matching infinities

*Blocked by: Task 1. Blocks: Task 4.*

Files owned: `src/harness/bundle/OutputComparison.hpp`,
`src/harness/bundle/OutputComparison.cpp`, `tests/TestOutputComparison.cpp`.

Split into two branch flips, each with its own red:

**3a — the float path.**
1. **Red first.** Add T1–T5, T7, T8 (§7.2). With the Task 1 stub still returning
   `ValidatorKind::ALLCLOSE`, T1, T5's accept-half and T7 fail as assertions; T2, T3, T4 and
   T8 pass for the wrong reason (record that in the checklist — they are regression guards,
   not the flip).
2. Change the factory to emit `ValidatorKind::ALLCLOSE_MATCHING_INFINITIES` and drop the stub
   comment.
3. Add the `case ValidatorKind::ALLCLOSE_MATCHING_INFINITIES:` to `makeValidator`
   (`OutputComparison.cpp:37-75`), calling
   `hipdnn_test_sdk::utilities::createAllCloseMatchingInfinitiesValidator(dataType,
   tolerance.atol, tolerance.rtol)`.
4. Leave `formatMismatchReport` alone (Q3). Leave `default: throw` at `:70-74` alone.

**3b — the unsupported-dtype path.**
1. **Red first.** T6 (§7.2) now fails by *throwing* out of `compareTensor` (`ASSERT_NO_THROW`
   fires), because step 3a added the case outside a `try`.
2. Wrap the new case in a `try/catch(const std::exception&)` of the same shape as the RMS one
   at `:49-68`. To avoid two copies of the NOT-APPLICABLE message, extract a file-local helper
   in the `.cpp` — e.g.

   ```cpp
   std::string validatorNotApplicable(const std::string& label,
                                      hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                                      const char* validatorName,
                                      const char* reason);
   ```

   — and call it from both the RMS and the new catch block. The rendered text must keep every
   element the existing message has, including the literal `[[validator_overrides]]` and the
   "Narrow that entry's 'tensors' glob" instruction, with only the validator name varying.
3. Update the `makeValidator` doc comment (`OutputComparison.hpp:131-137`) — it currently
   names RMS as the only glob-selectable kind with a dtype restriction. It now has two.

**Acceptance**: T1–T8 green; every existing `TestOutputComparison` test still green,
specifically `RmsOnAnUnsupportedDataTypeIsReportedNotThrown` (`:413`, asserts on `"counts"`,
`"INT32"` and `"validator_overrides"` — the helper extraction must not disturb it),
`AllcloseStillGradesIntegerOutputs` (`:442`) and `UnhandledValidatorKindIsRefused` (`:461`).

---

#### Task 4 — Generalise the TomlGuards translation (the gating, non-mechanical edit)

*Blocked by: Tasks 1 and 3. Blocks: Task 5.*

Files owned: `src/harness/TomlGuards.hpp`, `tests/TestTomlGuards.cpp`.

**Unowned-file sweep**: `tests/TestTomlGuards.cpp` is edited by this task and by no other. It
names the old helper at lines `:11`, `:52`, `:81`.

1. **Red first.** Rename the call sites in `TestTomlGuards.cpp` to
   `findTomlValidatorOverride` and rewrite T12/T13 (§7.2) to assert on
   `std::optional<ValidatorOverride>`. This fails to compile until step 2 lands — an
   acknowledged weak red (Risk R3); the substantive red for this task is T14, added at
   step 4.
2. Replace `findTomlRmsThreshold` (`TomlGuards.hpp:58-71`) with:

   ```cpp
   inline std::optional<ValidatorOverride>
       findTomlValidatorOverride(const std::string& testName, const std::string& tensorLabel)
   {
       if(testName.empty())
       {
           return std::nullopt;
       }
       return TestConfig::get().findValidatorOverride(testName, tensorLabel);
   }
   ```

   Carry over the existing doc comment's two load-bearing points — that it is shared by both
   verification harnesses so the selection cannot be described two ways, and that
   `tensorLabel` must be `bundle::tensorLabel(uid, name)` and never a raw tensor name — and
   restate the "nullopt means allclose" sentence in terms of *no matching entry* rather than
   *no RMS entry*.
3. Rewrite `gradingForTensor` (`TomlGuards.hpp:82-102`) as a switch over the selected kind.
   Shape:

   ```cpp
   inline bundle::ComparisonTolerance gradingForTensor(const std::string& testName,
                                                       const std::string& tensorLabel,
                                                       float atol,
                                                       float rtol)
   {
       if(const auto selected = findTomlValidatorOverride(testName, tensorLabel))
       {
           switch(selected->kind)
           {
           case ValidatorOverrideKind::RMS:
               HIPDNN_PLUGIN_LOG_INFO(... << ": rms, threshold=" << selected->rmsThreshold);
               return bundle::ComparisonTolerance::rms(selected->rmsThreshold);

           case ValidatorOverrideKind::ALLCLOSE_MATCHING_INFINITIES:
               applyTomlToleranceOverride(testName, atol, rtol);
               HIPDNN_PLUGIN_LOG_INFO(... << ": allclose_matching_infinities, atol=" << atol
                                          << " rtol=" << rtol);
               return bundle::ComparisonTolerance::allCloseMatchingInfinities(atol, rtol);

           case ValidatorOverrideKind::ALLCLOSE:
               break; // an explicit allclose entry takes the default path below

           default:
               throw std::invalid_argument("gradingForTensor: unhandled ValidatorOverrideKind");
           }
       }

       if(applyTomlToleranceOverride(testName, atol, rtol))
       {
           HIPDNN_PLUGIN_LOG_INFO(... << ": allclose, atol=" << atol << " rtol=" << rtol);
       }
       return bundle::ComparisonTolerance::allClose(atol, rtol);
   }
   ```

   Note the matching-infinities branch logs **unconditionally** once selected — mirroring RMS,
   which announces itself whether or not a tolerance override also matched. `<stdexcept>` must
   be added to the header's includes.
4. **Red for the substantive behaviour.** Add T14 (§7.2) — a direct
   `bundle::ComparisonTolerance::allCloseMatchingInfinities(...)` round-trip asserting kind,
   atol and rtol.

   **Measure T14's red by mutation, not by sequence.** The factory is wired in Task 3a step 2,
   so by the time this task runs T14 is green on arrival and no ordering of these steps can
   observe it red. Instead, inject the exact defect T14 exists to catch: transiently revert the
   factory to return `ValidatorKind::ALLCLOSE`, rebuild, observe T14 fail, then restore and
   re-confirm green. That proves T14 discriminates on this failure mode, which a merely
   sequenced red would prove only incidentally. Bound the window to one build, and make no
   report, handoff, or commit while the factory lies — the Task 1 stub rule at `:440-446`
   applies unchanged. Both observations go in the handoff.
5. Preserve the existing comment at `:73-81` explaining that the validator override is read
   *first* because it outranks atol/rtol.

6. **Run the §7.5 join check now** — this is the first point at which the whole chain exists.
   It is a required step of this task, not an optional extra, and its transcript is part of
   the handoff (§6.3). It also discharges Q1, which no unit test can.

**Acceptance**: T12, T13, T14 green; `GradingForTensorKeepsTheCallersToleranceWhenNoSettings`
(`TestTomlGuards.cpp:87`) still green — the no-config default is still `ALLCLOSE` with the
caller's atol/rtol, unchanged; **and the §7.5 join check observed green, with the scratch
config removed and `git status` clean.**

---

#### Task 5 — Documentation, and only documentation

*Blocked by: Tasks 2 and 4.*

Files owned: `dnn-providers/integration-tests/README.md`,
`dnn-providers/hip-kernel-provider/config/HIP_MLOPS_ENGINE.toml` (**comment block only**).

1. `README.md:433-450` — extend the `validator_overrides` bullet: `validator` is now
   `"allclose"`, `"allclose_matching_infinities"` or `"rms"`. State that
   `allclose_matching_infinities` grades exactly as allclose except that same-signed
   infinities in reference and device output compare equal; that NaN, opposite-signed
   infinities and finite-vs-infinite still fail; that it is for outputs whose correct value is
   infinite on both sides (an SDPA forward log-sum-exp row that is fully masked); that
   `rms_threshold` must be absent; and that it is defined for float/half/bfloat16/double only,
   with the same "narrow your glob" failure for an integer output that `"rms"` already has at
   `:441-443`.
2. `HIP_MLOPS_ENGINE.toml:25-33` — extend the `-- Validator Overrides --` comment block's
   `validator:` line to name all three spellings. **Add no `[[validator_overrides]]` entry to
   this or any other engine TOML** (§3.5 / §2.3).

**Acceptance**: no code change, no test change, **and no oracle run** — neither `README.md`
nor a TOML comment is read by the unit binary, so re-running the suite here would prove
nothing. If it is run anyway, record it as a formality, not as evidence. No red exists for
this task by construction (§7.3).

---

### 6.3 Handoff to the orchestrator

**Implementor reports**: the oracle command actually run and its exit code; the pass/fail
counts before and after each red/green flip; the §7.5 join-check transcript (command, exit
code, matched log lines, `git status --porcelain` output); the test-name-validator run; and
the log paths. It runs **no** git commands other than the `git status` confirmation in §7.5
step 4, and **no** build beyond the oracle and the join check.

**Orchestrator owns**, after review of the diff:

1. The commit (Implementors do not commit).
2. The separate mandatory clang-tidy step (§7.4) — `ENABLE_CLANG_TIDY=OFF` in the existing
   cache, so no build so far has covered it.
3. The pre-commit substitute (§7.4).
4. **The PR body.** This is a deliverable, not a risk note. It must:
   - carry the **JIRA ID on its own line**, and follow the `/hipdnn-pr-quality` body template
     exactly (section order, checklist format) — the repo copy at
     `projects/hipdnn/tools/ai/skills/hipdnn-pr-quality/SKILL.md` is the source of truth;
   - state that **the knob ships with no consumer**, and why (§2.4, §3.5);
   - state the **AC6 tension and the decision to land here anyway** (§2.5);
   - state that the end-to-end join was verified **once, by hand**, and is **not** in CI (R1,
     §7.5) — do not present it as test coverage;
   - contain **no** internal hostnames, cluster names or SLURM references; name the ASIC only
     (`gfx950`, `gfx1152`).

---

## 7. Testing Plan

### 7.1 Local oracle (§2b) — named, and runnable by the Implementor

> **The oracle is the `hipdnn_integration_tests_unit_tests` quick suite, run whole.** It is
> below end-to-end, runs on CPU only, takes ~6 s, and must be run before the Implementor
> reports. The compile-and-test suite does not count.

```bash
# 1. Build the target (from the worktree root)
cd C:/develop/claude_workspace/worktrees/rocm-libraries/ALMIOPEN-2610
ninja -C build hipdnn_integration_tests_unit_tests \
  > C:/develop/claude_workspace/WIP/worktrees/rocm-libraries/ALMIOPEN-2610/build.log 2>&1 \
  || { echo "Build failed:"; tail -100 C:/develop/claude_workspace/WIP/worktrees/rocm-libraries/ALMIOPEN-2610/build.log; false; }

# 2. Run the oracle
cd C:/develop/claude_workspace/worktrees/rocm-libraries/ALMIOPEN-2610/build/dnn-providers/integration-tests
ctest -R hipdnn_integration_tests_unit_tests_quick_suite --output-on-failure \
  > C:/develop/claude_workspace/WIP/worktrees/rocm-libraries/ALMIOPEN-2610/oracle.log 2>&1 \
  || { echo "Oracle failed:"; tail -100 C:/develop/claude_workspace/WIP/worktrees/rocm-libraries/ALMIOPEN-2610/oracle.log; false; }
```

The `build/` directory already exists and is configured (§3.6); no reconfigure is needed. If
one is ever needed, it must carry `-DHIPDNN_ENABLE_KERNEL_INGESTOR=ON` to match the cache.

**While iterating on a single red**, the binary may be run directly — but **never with a
narrow `--gtest_filter`** (§3.6 gotcha: four `TestTomlGuards` tests fail unless
`TestTestConfig.cpp`'s suite runs first in the same process). Run it whole:

```bash
cd C:/develop/claude_workspace/worktrees/rocm-libraries/ALMIOPEN-2610/build
./bin/hipdnn_integration_tests_unit_tests.exe 2>&1 | tail -40
```

Baseline verified green 2026-09-23: `1/1 Test #5 ... Passed 6.17 sec`.

### 7.2 New tests

All run **on the local host CPU**, in `bin/hipdnn_integration_tests_unit_tests.exe`, built by
`ninja -C build hipdnn_integration_tests_unit_tests`. No GPU, no device, no bundle fixture —
`tests/CMakeLists.txt:4-8` makes "these unit tests never reach an engine or a device" a
link-time guarantee. Tensors are built in-process from the existing `K_GRAPH` flatbuffer
(`TestOutputComparison.cpp:43-71`), so nothing is borrowed from another subsystem.

`tests/TestOutputComparison.cpp` — suite `TestOutputComparison`:

| # | Test | Runs on / against | Discharges |
|---|------|-------------------|------------|
| T1 | `AllcloseMatchingInfinitiesAcceptsSameSignedInfinities` | host CPU / `floatTensor3(K_UID_B, -inf, 1.0f, -inf)` on both sides | The new kind accepts; plain allclose on the *same* pair still rejects. Both assertions in one test, so the disagreement is the pinned fact. |
| T2 | `AllcloseMatchingInfinitiesStillRejectsOppositeSignedInfinities` | host CPU / ref `-inf` vs impl `+inf` | Sign is part of the match (`CpuFpReferenceValidation.hpp:80`) |
| T3 | `AllcloseMatchingInfinitiesStillRejectsNaN` | host CPU / ref NaN vs impl NaN | NaN is not an infinity and is still a failure (`:85`) |
| T4 | `AllcloseMatchingInfinitiesStillRejectsFiniteVersusInfinite` | host CPU / ref `-inf` vs impl `-1e30f` | Finite-vs-infinite is still a failure |
| T5 | `AllcloseMatchingInfinitiesStillGradesFiniteElementsByAtolRtol` | host CPU / `(-inf, 1.0f, 2.0f)` vs `(-inf, 1.1f, 2.0f)` | Same tensor fails at atol=rtol=0 and passes at a tolerance covering 0.1 — finite elements are still graded, and the infinity does not mask them |
| T6 | `AllcloseMatchingInfinitiesOnAnUnsupportedDataTypeIsReportedNotThrown` | host CPU / `intTensor(K_UID_INT, 7)` on both sides (INT32, equal) | `ASSERT_NO_THROW`; the returned report names `counts`, `INT32`, `validator_overrides`, the validator's own name, **and the literal `Narrow that entry's 'tensors' glob`** — the last of these is what stops the Task 3b helper extraction from silently dropping the remediation sentence (R4). Mirrors `:413`. |
| T7 | `AllcloseMatchingInfinitiesIsChosenPerTensor` | host CPU / `compareOutputs` over `{K_UID_A, K_UID_B}`, lookup returning the new kind for `"uid=4"` and plain allclose for `"y_out"`, both holding `-inf` | Per-tensor selection through the `ToleranceLookup` label — what a `tensors` glob drives. Mirrors `:354`. |
| T8 | `AllcloseMatchingInfinitiesFailureStillReportsAtolRtol` | host CPU / a finite drift under the new kind | The report prints `atol=` and does **not** print `relative RMS` — pins Q3's no-change |

`tests/TestTestSettings.cpp` — suite `TestSettingsValidatorOverrides`:

| # | Test | Runs on / against | Discharges |
|---|------|-------------------|------------|
| T9 | `ParsesAllcloseMatchingInfinities` | host CPU / `TempTomlFile` with `filters`+`tensors` globs and `validator = "allclose_matching_infinities"`, no threshold | The name parses to `ValidatorOverrideKind::ALLCLOSE_MATCHING_INFINITIES`, and selection still requires **both** globs to match. Mirrors `:336`. |
| T10 | `AllcloseMatchingInfinitiesLaterEntryTakesPrecedence` | host CPU / an `rms` entry followed by a narrower `allclose_matching_infinities` entry | Precedence is unchanged by the new kind. Mirrors `:377`. |
| T11 | `ThrowsOnAllcloseMatchingInfinitiesWithRmsThreshold` | host CPU / the same entry carrying `rms_threshold = 1e-4` | The shared "does not say what it means" guard covers the third name too (Q2). Mirrors `:473`. |

`tests/TestTomlGuards.cpp` — suite `TestTomlGuards`:

| # | Test | Runs on / against | Discharges |
|---|------|-------------------|------------|
| T12 | `FindTomlValidatorOverrideReturnsNulloptForEmptyName` | host CPU / uninitialized-name early return | Renamed from `FindTomlRmsThresholdReturnsNulloptForEmptyName` (`:50`); same contract on the new return type |
| T13 | `FindTomlValidatorOverrideReturnsNulloptWhenNoSettings` | host CPU / `TestConfig` initialized without a settings file | Renamed from `:79`; keeps the "absence of an entry is what expresses the default" comment |
| T14 | `AllCloseMatchingInfinitiesToleranceCarriesAtolAndRtol` | host CPU / `bundle::ComparisonTolerance::allCloseMatchingInfinities(1e-3f, 2e-3f)` | The new kind carries real atol/rtol and is not zeroed like `rms` — constraint §2.2(6), and Q1's tolerance path |

**Existing tests that must stay green** (regression guards, not new coverage):
`RmsOnAnUnsupportedDataTypeIsReportedNotThrown`, `AllcloseStillGradesIntegerOutputs`,
`UnhandledValidatorKindIsRefused`, `ValidatorKindIsChosenPerTensor`,
`RmsFailureReportsItsThresholdNotAtolRtol`, `ThrowsOnUnknownValidatorKind`,
`ThrowsOnAllcloseWithRmsThreshold`, `LaterEntriesTakePrecedence`,
`GradingForTensorKeepsTheCallersToleranceWhenNoSettings`.

### 7.3 Tasks with no red, by construction

- **Task 1** is additive-only (two enumerators and a stubbed factory nothing calls). There is
  no behaviour to fail.
- **Task 5** is documentation. There is no behaviour to fail.

Do not invent a red for either.

### 7.4 Gates outside the oracle

| Gate | Owner | Command |
|------|-------|---------|
| Local oracle | Implementor | §7.1 |
| Join check (one-off) | Implementor | §7.5 |
| **Test-name validation** | Implementor | **Not covered by the oracle.** ctest #4 `hipdnn-integration-tests_test_name_validation` (label `pre-commit`) runs `scripts/test_name_validator.py --strict` and sits outside `-R …_quick_suite`. All 14 proposed names conform to `test_name_validator.py:19-38`, but run it before reporting: `ctest -R hipdnn-integration-tests_test_name_validation` from `build/dnn-providers/integration-tests`. |
| Pre-commit | Orchestrator | `bash "$(git rev-parse --git-common-dir)/hooks/pre-commit"` — **never** `pre-commit run` |
| clang-tidy (C++) | Orchestrator | Separate mandatory step. `ENABLE_CLANG_TIDY=OFF` in the existing `build/` cache (§3.6), so the oracle build does **not** cover it. Four of the five edited source files are headers — see `feedback_clang_tidy_headers.md`. |
| Comment scrub | — | **SUSPENDED** per CLAUDE.md §4. Not run. |

**Suite-naming constraint (not a gate, but it fails silently).** `ctest -N -V` shows Test #5
carries `--gtest_filter=*-*DISABLED*:Standard*:Comprehensive*:Full*`. The exclusions are
negative and none of the 14 proposed names begins with `Standard`, `Comprehensive` or `Full`,
so the plan is safe as written — but a future test **suite** named with one of those prefixes
would be dropped from the oracle with no error and no count change. Keep new suites in
`TestOutputComparison` / `TestSettingsValidatorOverrides` / `TestTomlGuards`.

---

### 7.5 Join check — one-off manual verification that TOML reaches the new validator

**Owned by the Implementor. Run once, after Task 4. Mitigates R1 and discharges Q1.**

This is the user's Decision 2 of 2026-09-23. It is a hand-run demonstration, **not** a test
that ships and **not** regression coverage.

#### Method, and why this one

The discriminator is the `Validator override applied for … : <validator>, atol=… rtol=…` line
that `gradingForTensor` emits itself (`TomlGuards.hpp`, gated at runtime by
`HIPDNN_LOG_LEVEL=info` — `PluginLogging.hpp:86-94`). Observing that line naming
`allclose_matching_infinities` is a **direct** observation of the uncovered glue, and the
`atol=`/`rtol=` it prints answers Q1 in the same run.

The review suggested an integer-dtype tensor as the discriminator (new kind throws → NOT
APPLICABLE report; allclose grades integers fine). That works at unit level — it is what T6
pins — but it is **not the cleaner option here**: it proves the *branch*, not the *join*, it
requires finding an integer-typed output in a locally-runnable suite, and it reports a failure
rather than a success, so a mis-wired join and a correctly-wired one both end in red. The log
line discriminates positively, needs no integer output, and carries Q1's answer for free. Use
it. (T6 retains the integer case at unit level, where it belongs.)

**No tracked file is edited.** `--test-config <path>` (`src/main.cpp:189-199`) fully replaces
the engine config, so the scratch TOML lives outside the repository entirely. The shipping
prohibition (§2.3) is therefore structural, not a matter of remembering to revert.

#### Hardware

The local GPU is **gfx1152** (AMD Radeon 860M). **gfx950 is not locally reachable** — it is
SLURM-only, and the gfx950 dense engine arch-prunes before the matcher, so its bundles cannot
run here. **The join check needs none of gfx950, SDPA, or an infinity** — only a test whose
output tensors are graded through `gradingForTensor`. The layernorm path qualifies and runs
locally.

#### Verified runnable, 2026-09-23 (all four probes actually executed)

| Probe | Command | Observed |
|-------|---------|----------|
| Suite is locally runnable | `ctest -R hip-kernel-provider-external-integration_quick_suite` | **Passed, 366 s**, exit 0, on gfx1152 |
| Single case is fast | the one-test filter below, committed config | 1 ran, 1 passed, a few seconds |
| Control (scratch config works, label targeting works, Q1 observable) | scratch TOML, `validator = "rms"` on `*::DX` + a `[[tolerance_overrides]]` entry | `… tensor LayernormBackward_0::DX: rms, threshold=0.0001` **and** `… DSCALE: allclose, atol=0.125 rtol=0.25` |
| **Red half, today** | the same scratch TOML with `validator = "allclose_matching_infinities"` | **exit 1**, `Fatal error: TestSettings: [[validator_overrides]] entry has unknown validator 'allclose_matching_infinities' (expected "allclose" or "rms")` |

The red half is already confirmed. After implementation the same command must go green.

#### Checklist for the Implementor

1. **Write the scratch config outside the repo** — `C:/develop/claude_workspace/WIP/worktrees/rocm-libraries/ALMIOPEN-2610/joincheck/joincheck.toml`:

   ```toml
   [meta]
   version = 1

   # Present so the logged atol/rtol answers Q1: does a tolerance override still
   # apply when the matching-infinities validator is selected?
   [[tolerance_overrides]]
   filters = ["*LayernormBackwardPure4DFp32*"]
   atol = 0.125
   rtol = 0.25

   [[validator_overrides]]
   filters   = ["*LayernormBackwardPure4DFp32*"]
   tensors   = ["*::DX"]
   validator = "allclose_matching_infinities"
   ```

2. **Run it** (from `build/`; output is small, but redirect anyway):

   ```bash
   cd C:/develop/claude_workspace/worktrees/rocm-libraries/ALMIOPEN-2610/build
   HIPDNN_LOG_LEVEL=info ./bin/hipdnn_integration_tests.exe \
     --test-article "./bin/hipdnn_plugins/engines/hip_kernel_provider.dll" \
     --test-engine HIP_MLOPS_ENGINE \
     --test-config "C:/develop/claude_workspace/WIP/worktrees/rocm-libraries/ALMIOPEN-2610/joincheck/joincheck.toml" \
     --gtest_filter="Quick/IntegrationGpuLayernormBackwardPure4DFp32.Correctness/0" \
     > C:/develop/claude_workspace/WIP/worktrees/rocm-libraries/ALMIOPEN-2610/joincheck.log 2>&1
   echo "exit=$?"
   grep -i "override applied" C:/develop/claude_workspace/WIP/worktrees/rocm-libraries/ALMIOPEN-2610/joincheck.log
   ```

3. **Required observables** — all four, or the check has not passed:
   - exit code **0** (the config loads; contrast the red half above, which exits 1);
   - a line reading `… tensor LayernormBackward_0::DX: allclose_matching_infinities, atol=0.125 rtol=0.25`
     — the validator name proves the **join**, and `atol=0.125 rtol=0.25` (not the default
     tolerance) proves **Q1**;
   - `LayernormBackward_0::DSCALE` and `::DBIAS` still logged as plain `allclose` — the
     `tensors` glob selected per tensor and did not spread;
   - the test still **passes** — the new validator grades real finite data correctly, not just
     infinities.
4. **Delete the scratch file and the log**, and confirm the tree is clean:

   ```bash
   rm -rf C:/develop/claude_workspace/WIP/worktrees/rocm-libraries/ALMIOPEN-2610/joincheck
   cd C:/develop/claude_workspace/worktrees/rocm-libraries/ALMIOPEN-2610
   git status --porcelain
   ```

   `git status` must show no engine `.toml` as modified. Because the scratch file was never
   inside the worktree, this is a confirmation step rather than a revert — but **run it and
   paste the output in the report**. The orchestrator must not commit until it has seen a
   clean `git status` for every `config/*.toml`.
5. **Report**: the exact command, exit code, the matched log lines, and the `git status`
   output. State explicitly that this was a one-off manual check and is not in CI.

## 8. Risks and Considerations

**R1 — `gradingForTensor`'s new branch will have NO coverage of any kind, unlike RMS.**
This is the plan's most serious weakness. An earlier draft claimed the RMS branch was "equally
uncovered"; **that claim was false and has been removed.** The corrected position:

- `TestConfig` is a one-shot singleton (`TestTestConfig.cpp:222-226`) initialized in the unit
  binary *without* a settings file, so no unit test can put a loaded TOML in front of
  `gradingForTensor`. True for RMS and for the new kind alike — **neither has a unit-test
  oracle for the glue.**
- **But RMS has live integration coverage, and the new kind will have none.**
  `HIP_MLOPS_ENGINE.toml:45-49, 55-59, 61-65` carry three committed `validator = "rms"`
  entries against `*LayernormBackward*`. Every run of those tests drives the full chain — TOML
  text → `ValidatorOverrideKind::RMS` → `ComparisonTolerance::rms` → `createRmsValidator`.
  **Verified locally 2026-09-23** on gfx1152 by running
  `Quick/IntegrationGpuLayernormBackwardPure4DFp32.Correctness/0` with `HIPDNN_LOG_LEVEL=info`;
  `gradingForTensor` announced itself twice:

  ```
  Validator override applied for Quick/IntegrationGpuLayernormBackwardPure4DFp32.Correctness/0 tensor LayernormBackward_0::DSCALE: rms, threshold=0.0001
  Validator override applied for Quick/IntegrationGpuLayernormBackwardPure4DFp32.Correctness/0 tensor LayernormBackward_0::DBIAS: rms, threshold=0.0001
  ```

  RMS lacks a *unit*-test oracle. It does not lack *coverage*. The new kind lacks both, because
  §2.3's shipping prohibition means no committed TOML will ever select it.
- **Covered by the halves only**: TOML text → `ValidatorOverrideKind` by T9/T10/T11 against a
  directly constructed `TestSettings`; `ComparisonTolerance` → validator → verdict by T1–T8.
  The glue between them, and Q1's tolerance-resolution line, are not covered by any automated
  test and **will not be**.
- **Mitigation (user decision, 2026-09-23): the §7.5 join check.** A one-off manual run with a
  scratch config proves the chain joins at least once. It is a **hand-run demonstration, not a
  regression test** — it runs once, is never committed, and nothing re-runs it. Do not describe
  it in the PR as test coverage.
- **Not attempted here**: making `gradingForTensor` a pure function over an injected
  `std::optional<ValidatorOverride>` would close this properly, but that is a new seam nobody
  signed off, and it routes to Design mode rather than into this plan. *Recommend it as a
  follow-up, tracked against ALMIOPEN-2446 per §2.5 — it would improve RMS's testability too.*
  Note this is unrelated to the `BundleReferenceValidationHarness` behaviour, which is settled
  policy and must not be filed (§2.3).

**R2 — the `default: throw` in `gradingForTensor` is unreachable and untested.** The parser
produces only the three enumerators it knows, so nothing can drive the default. It is
nonetheless the right guard (Q5), and it matches `makeValidator:70-74`, whose analogous throw
*is* testable only because `ValidatorKind` is reachable by cast from a test
(`UnhandledValidatorKindIsRefused`, `:461`). Accept the asymmetry; do not manufacture a cast
test through the `TestConfig` singleton to reach it.

**R3 — Task 4 step 1's red is a compile error, which is a weak oracle.** Renaming the helper
cannot fail as an assertion; it proves a symbol moved, not that behaviour is right. That is
why Task 4 adds T14 as its substantive red. Do not treat the rename's compile failure as the
task's red.

**R4 — the helper extraction in Task 3b can silently weaken an existing message.**
`RmsOnAnUnsupportedDataTypeIsReportedNotThrown` (`:413`) asserts only on `"counts"`, `"INT32"`
and `"validator_overrides"`. A refactor could drop the "Narrow that entry's 'tensors' glob"
sentence — the only part of the message that tells an operator what to *do* — and stay green.
**Mitigation: T6 asserts on that literal sentence** (§7.2). A by-hand diff is not a gate and
is not an acceptable substitute.

**R5 — `-Werror -Wswitch-default` will reject any switch you add without a `default:`**
(§3.6). This bites the moment a `switch(ValidatorOverrideKind)` is written. It is why Q5 has
only two live options.

**R6 — the new kind must never become reachable by default.** The single guard is that
`ComparisonTolerance`'s member default stays `ValidatorKind::ALLCLOSE`
(`OutputComparison.hpp:64`) and that `gradingForTensor`'s no-entry path still returns
`allClose`. `GradingForTensorKeepsTheCallersToleranceWhenNoSettings` is the regression guard;
it must not be weakened.

**R7 — `references/hipdnn.md` names a `hipdnn-integration-tests-unit-check` target that does
not exist in this configuration** (§3.6). An Implementor following the reference doc will get
`ninja: error: unknown target` and may fall back to a tier target that pulls in GPU tests.
The oracle in §7.1 is the authority. *Recommend a follow-up correction to that reference doc;
not in this ticket's scope.*

**R8 — the PR body carries three disclosures that no code gate can enforce.** No consumer
(§2.4), the AC6 tension (§2.5), and the hand-run-only join check (R1). A reviewer who is told
none of these will reasonably ask which test uses the knob, and the honest answer is too long
for a review thread. **This is now an assigned deliverable, not a risk** — see §6.3 item 4,
which also pins the `/hipdnn-pr-quality` template, the JIRA-ID line, and the
`feedback_pr_no_internal_tools.md` rule (name the ASIC only; no cluster hostnames or SLURM
references). Retained here only so the risk register points at the owner.

**Compatibility**: purely additive. Every existing engine TOML parses unchanged; both
existing enumerators keep their values; both existing `ComparisonTolerance` factories use
aggregate initialization and compile unchanged when an enumerator is appended. No serialized
format carries either enum (grep of `ValidatorKind|ValidatorOverrideKind|ComparisonTolerance::`
across the tree shows 20 sites, all in-process). No performance consideration — the validator
is selected once per output tensor.
