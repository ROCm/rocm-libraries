# Design — SHADOW Capture as Canonical Reference

**Status:** proposed v5 (2026-06-01, post-third-review-verification)
**Supersedes:** Approach A (`build_non_cms_reference`, `_pre_cms_state` snapshot, the inline shape-(a) branch at `KernelWriter.py:5742-5774`, most of the nyb5 / y391 / l1l6 / hdem / m7o5 work stack)
**Implements:** the right reference for "did the CMS scheduler do its job"
**Companion beads:**
- `rocm-libraries-nmsx` (reopened) — three SHADOW capture-window / scope / walk-coverage fixes. The earlier "ltnz idMap inversion" framing was wrong — verified directly that SHADOW already uses `build_idmap` + `invert_idmap_to_id_to_category` at `KernelWriter.py:4720, 4735, 4751`. The three defects are not parallel-implementation drift; they are capture-window timing (LCC), count truncation (PLR1 packs), and hand-rolled tag emission inside `_makeSubIterSchedule` (LRS/LWS pointer-math).

## 1. Problem

The validator's job is **"did the CMS scheduler order this kernel's instructions correctly?"** The current reference (Approach A, `build_non_cms_reference`) builds an *entirely different kernel* with `UseCustomMainLoopSchedule=0` re-derived from a `pre_cms_state` snapshot. That flips `doFullPackCodePrefetch` and other code-path gates, producing a **different instruction set** (e.g. BPG#11: 20 extra `v_mov_b64` saves on the reference vs the subject — see `rocm-libraries-j4qm`). Comparisons confound scheduler-quality with codegen-branch divergence.

The right reference shares codegen state with the CMS subject and differs only in scheduler choice. This is what the legacy SHADOW path (`_captureDefaultSchedule`) already produces — and empirical evidence (2026-06-01) confirms it produces it deterministically and correctly excludes the j4qm codegen-noise.

### Reconciliation with the 2026-05-26 "real vs real" directive

Verbatim user directive (`71HW_DECOMPOSITION.md:59`, 2026-05-26):

> *"I want us to move off the shadow path. It must be real vs real. This must be the next thing we directly work towards."*

**v5 is a reversal of the "move off the shadow path" part of that directive, with explicit acknowledgment.** The reasoning:

The 2026-05-26 directive was issued in response to SHADOW's known *implementation defects* (LCC absence, missing instruction propagation). The interpretation at the time was structural: move off the SHADOW *path* and build a second real kernel via Approach A. That interpretation introduced a different and worse problem: a second CMS=0 kernel yields a different *instruction set* than CMS=1 (different `doFullPackCodePrefetch`, different T-reg allocation, different pack chain). Comparing them tells you the two kernels differ in codegen-branch elections — which is not the validator's question.

The validator's question is narrower and more specific: **did the user correctly schedule the instructions from CMS=1, vs. what the default scheduler would do if given the exact same instructions?** Same instruction set on both sides is a precondition. A second CMS=0 kernel doesn't satisfy it; SHADOW (operating on the CMS-derived codegen state) does.

So what changed concretely: the 2026-05-26 directive was *right about the goal* (escape SHADOW's defects, get a trustworthy reference) but *wrong about the mechanism* (move off SHADOW entirely). v5 keeps SHADOW as the canonical reference and addresses its defects directly via nmsx. This is a path-reversal relative to 2026-05-26 — owning it explicitly here rather than hand-waving.

## 1.5. Scope of the validator (clarifying, not narrowing)

The validator answers **one question**: given the kernel's emitted instructions, did the CMS scheduler order them correctly? Specifically: are the issue cycles, wait-count placements, and barrier placements legal under the per-arch quad-cycle / MFMA-timing / pair-formation rules, and at least as well-formed as what the default scheduler would produce on the same instructions?

The validator does **not** answer (and was never designed to):
- Whether the kernel's codegen-branch elections are sound (e.g. is `doFullPackCodePrefetch=False` safe under this kernel's VGPR budget — the j4qm class). Codegen correctness is a different layer's problem, not the mainloop-scheduler validator's.
- Whether individual emitted instructions have correct operands (e.g. wrong register on an MFMA). That's the codegen emitter's contract to maintain.

Approach A's accidental "catching" of codegen-branch divergence (the j4qm noise) was a bug of the wrong reference shape — not a feature being lost. The new reference correctly excludes it.

## 2. The right reference

**Subject** (unchanged): CMS=1 kernel built with the per-tile, user-authored OptSchedule from files like `Tensile/Components/CustomSchedule/gfx950/_128x160x64_TF32.py`.

**Reference (new):** the capture already produced by the existing `_captureDefaultSchedule` machinery at `Tensile/KernelWriter.py:4697-4784` and `:3922-3935`. It runs the real default scheduler (`makeSchedule` → `SIA3.schedIntoIteration` → `_makeSubIterSchedule`) over the **same writer state** that produced the CMS subject, mid-build, and stashes the result at `writer._last_default_capture` (property at `KernelWriter.py:452-456`).

The "SHADOW path" name is misleading — its capture is real, taken from the real default scheduler operating on the real CMS-derived codegen state. It was renamed "SHADOW" historically because it's not assembled to runnable code, but the capture itself records exactly what the default scheduler chose. That's what the validator needs.

## 3. Comparison contract

Empirically verified (sonnet experiment 2026-06-01, results in `6hk3_artifacts/SHADOW_VIABILITY_EXPERIMENT.md`):

- **Deterministic:** two separate Tensile process invocations of BPG#11 produce byte-identical SHADOW captures (215,872 bytes each).
- **Stable intra-process:** `_initKernel:6738` creates a fresh `CaptureContext()` per kernel; `finally:` at `:5723` provides secondary reset. No cross-kernel state leakage.
- **Correctly matches CMS subject's data-flow instruction set:** SHADOW emits 0 v_mov_b64 in mainloop (matching CMS subject's 0), proving the j4qm codegen-branch noise is correctly excluded. (Approach A by contrast emits 20.)
- **Better categorization than Approach A:** Approach A's reference has 200 UNKNOWN-classified instructions (144 v_cvt_pk_bf16_f32, 36 ds_read_b128, 20 v_mov_b64). SHADOW has none — its capture goes through `capture_id_to_cat` which uses the canonical idMap categories.

With the SHADOW capture as reference (post-nmsx):
- **Per-category counts** are equal by construction on data-flow categories (LR/LW/GR/MFMA/CVT_PACK/MIDDLE_PACK), modulo SYNC/SNOP which are scheduler-inserted.
- **Identity uses `(canonical_render, source_module_id)` tuple-set equality** (not Python object identity — see §3 caveat below).
- **Edge keys** compare cleanly under the tuple-set identity. T0/X0 false-positives disappear because both sides see the same register names (same codegen).
- **Wait-coverage validation** runs against the CMS subject and now has a meaningful pass criterion.

Identity-tuple caveat: LR/Pack per-iter Modules are constructed unnamed at `KernelWriter.py:4218-4221` (`Module()` with no name argument), so their leaves have `source_module_id=None`. For these, the tuple-set degenerates to `canonical_render`-multiset equality — which is sufficient *because the same codegen runs on both sides*, so both sides emit the same canonical_render strings.

## 4. Implementation

### Phase 0 — Deletion audit (must precede Phase 4)

Per the prior viability investigation: NOT all symbols on the v2 deletion list are Approach-A-only. Confirmed audit findings:

| Symbol | Classification | Safe to delete? |
|---|---|---|
| `build_non_cms_reference` | Approach-A-only at call sites; 1 debug-tool consumer (`_dump_carveout_assembly.py:187-228`) | Yes; migrate or break the carve-out tool separately |
| `pre_cms_state` / `_pre_cms_state` | Approach-A-only | Yes |
| `_make_solution` | **Load-bearing** (`MOCK_AUDIT.md:162`, 8 sites in `test_ScheduleCapture.py`, also `_dump_carveout_assembly.py`) | **NO** — keep |
| `enable_capture_non_cms_build` / `_captureNonCmsBuild` | Approach-A-only (xj16 block at `:5732` is replaced by Phase 2 wiring) | Yes |
| `_last_default_capture` | **Load-bearing** (`cms_from_default.py:123,186` — the schedule-conversion CLI tool) | **NO** — keep, in fact this is THE accessor the new design uses |

**Acceptance:** the per-symbol audit table above is checked-in (with the design doc) and verified by `git grep` immediately before Phase 4. No symbol's classification changed since audit.

### Phase 1 — Land three SHADOW capture-window / scope / walk-coverage fixes (`rocm-libraries-nmsx`)

**Important framing correction from v4:** v4 framed Phase 1 as "eliminate SHADOW's parallel categorization via idMap inversion." Direct code verification (third opus reviewer + independent verifier) confirmed this was factually wrong — SHADOW already uses `build_idmap` (`KernelWriter.py:4735`) and `invert_idmap_to_id_to_category` (`KernelWriter.py:4751`). There is no parallel walk to eliminate. The three documented SHADOW defects (LCC absence, PLR1 packs missing, LRS/LWS schema mismatch) are NOT parallel-implementation drift.

What they actually are:

#### Fix 1 — LCC absence (capture-window timing)

`KernelWriter.py:4747` passes `loopCounterCode=Module()` (empty) into `build_idmap` with the comment *"LCC items are added by customMainLoopSchedule, not SIA3"*. SHADOW finalizes BEFORE `customMainLoopSchedule` adds the two LCC instructions (`s_sub_u32` + `s_cmp_eq_i32`) at `closeLoop` emission (`:4865-4868`). The fix is to extend SHADOW's capture window so it observes the LCC items after they're added, OR to inject LCC items into the capture builder after `closeLoop` runs.

This is not a categorization issue. Categorization is already correct via the existing idMap inversion at `:4751`. The defect is *what's visible at capture time*.

#### Fix 2 — Missing PLR1 packs (scope / count truncation)

`KernelWriter.py:4736` invokes `build_idmap(num_loop_iter=len(LRCodeAAllIters), ...)`. At the moment SHADOW runs, `LRCodeAAllIters` is populated up to the current iter index, not the full loop. Subiter packs beyond that aren't in the SHADOW capture. The fix is to capture (or re-capture) after later subiters populate `LRCodeAAllIters`/`PackCodeAAllIters`, or to drive the capture from the full per-iter ranges rather than the populated-so-far range.

Also a scope/window issue, not categorization.

#### Fix 3 — LRS/LWS schema mismatch (hand-rolled tag emission)

Per the verification reviewer: `KernelWriter.py:1040-1045` (inside `_makeSubIterSchedule`) hard-codes `"LRS"` / `"LWS"` tags for pointer-math leaves. `build_idmap` at `ScheduleCapture.py:1045-1048` canonically uses per-side `"LRSA"` / `"LRSB"` / `"LWSA"` / `"LWSB"` tags for the corresponding swap-iter modules. The two sites are different Python objects (per `ScheduleCapture.py:1033-1039`'s comment), so the swap-iter portion is already correct via idMap inversion; only the pointer-math portion needs the manual split.

The fix is to split the pointer-math tag emission at `:1040-1045` by inspecting which Module the leaf came from (LRSwapA → "LRSA", etc.).

This is the closest of the three fixes to a "parallel implementation" — it's a hand-rolled tag emission at a single site. Single site, single fix, not a centralized enumeration.

#### Fail-loud contract

For all leaves: if the SHADOW capture encounters a leaf with neither an idMap entry nor a registered class-name (per `InstructionCategory._CLASS_NAME_TO_CATEGORY`), raise `CaptureCategoryMissingError` naming the class, body, and surrounding context. No silent "default to UNKNOWN." SIA3 adding a new control op class fails the build immediately rather than slipping through as a downstream comparison defect.

#### Acceptance

After all three fixes land:
1. SHADOW capture's per-category counts on BPG#11 match CMS subject's (modulo SYNC/SNOP — those are scheduler-inserted and legitimately differ)
2. `verify_correct_number_of_instructions(scheduleInfo, idMap)` passes against the CMS-side scheduleInfo
3. xj16 inline assertion at `KernelWriter.py:5818-5850` with `ctx.default = self._last_default_capture` runs to completion on BPG#11

#### Why this isn't the m7o5 pattern

m7o5's `_CMS_FRAMEWORK_DERIVED_DEFAULTS` was a centralized enumeration of CMS-derived flags. New framework-derived flags silently broke it because the scrub list was inseparable from per-flag knowledge.

These three fixes target three structurally distinct sites: capture-window scope (`:4747`), per-iter count parameterization (`:4736`), and one site of hand-rolled tag emission (`:1040-1045`). They are not a list of patches over a single drifting parallel walk — they are three independent locations each with its own principled justification. Adding a new SHADOW defect later (which the third reviewer warned about) would surface as a separate architectural issue, not as growth of this list.

### Phase 2 — Wire SHADOW as `ctx.default`

At `Tensile/KernelWriter.py:5772` (the inline CMS-assertion site), replace:
```python
ctx.default = build_non_cms_reference(kernel, self.assembler, isaInfoMap)
```
with the SHADOW capture, which is already populated by this point in the build:
```python
ctx.default = self._last_default_capture
```

`isValid` and `verify_correct_number_of_instructions` already run earlier. No other validator-layer changes needed.

**Acceptance:** the BPG#11 reproducer (`6hk3_tf32_128x160x64_tn.yaml`) builds successfully and the validator runs to completion. Expected outcome: zero residuals (the j4qm-style asymmetry vanishes because instructions are identical on both sides).

### Phase 3 — Validate against the CMS test surface, HARD GO/NO-GO GATE

**Note:** v4 referenced `_3ija_residual_triage_runner.py` here; that runner was deleted in commit `5fbe10db5e4` (bead `2bww`). Phase 3 uses the current CMS test surface instead.

Build the gfx950 CMS test surface (`Tensile/Tests/common/gemm/gfx950/custom_mainloop_scheduling_*.yaml` — TF32, 16-bit, and any other CMS-enabled YAML test files) with xj16 validation enabled (`ctx.default = self._last_default_capture` from Phase 2). For every CMS=1 kernel that builds:

1. `verify_correct_number_of_instructions(scheduleInfo, idMap)` must pass (already in place at `CMSValidator.py:4702`)
2. `compare_graphs(ref=SHADOW, subj=CMS)` must return empty failure list
3. `validate_edge_wait_coverage(subj=CMS)` must return empty failure list

Each failure classifies as one of:
- **resolved** — was an Approach-A artifact; doesn't reproduce under the SHADOW reference. Document the proof in commit message or memo.
- **real CMS bug** — the CMS schedule has a genuine wait-coverage or edge-coverage defect that SHADOW proves is fixable. Open a bead.
- **out-of-validator-scope** — surface is a codegen-correctness concern (e.g. malformed operands, codegen-branch election problems). Document why it's out of scope per §1.5 and close the bead as misfiled or move it to a codegen-quality tracker.

**Hard gate:** Phase 3 PASSES only when ZERO residuals are in "needs investigation" status. Every residual either has documented resolution, an open follow-up bead, or an out-of-validator-scope classification. Phase 4 cannot land otherwise.

**Fixture-coverage requirement (carried from Phase 1's multi-fixture concern):** the test surface chosen must collectively exercise every code path in `build_idmap` (`ScheduleCapture.py:1042-1078`) — specifically each optional-input branch (sparse-MX `LRCodeMXSA`/`LRCodeMXSB`/`LRCodeMetadata`), each dispatcher-time conditional, and each per-iter category split. Enumerate `build_idmap`'s branches first; pick or add fixtures to cover any uncovered branch.

### Phase 4 — Retire Approach A (blocked on Phase 0 + Phase 3)

Delete (subject to Phase 0 audit):
- `Tensile/Components/CustomSchedule/approach_a.py`
- `Solution._pre_cms_state` + `pre_cms_state()` accessor + the snapshot site at `Solution.py:1999`
- `enable_capture_non_cms_build` + `_captureNonCmsBuild` flag handling
- Tests pinned to Approach-A semantics: `test_approach_a_non_cms_reference.py` and friends
- The inline shape-(a) branch at `KernelWriter.py:5742-5774`

Keep (confirmed load-bearing per Phase 0):
- `_make_solution` (`cms_test_utils`) — 8 test sites + `_dump_carveout_assembly.py`
- `_last_default_capture` accessor — used by `cms_from_default.py` AND now the canonical reference accessor
- `_captureDefaultSchedule` machinery — IS the new reference (after nmsx fixes)
- `invert_idmap_to_id_to_category` at `ScheduleCapture.py:1081` — already in use at `KernelWriter.py:4751`; keep
- `verify_correct_number_of_instructions` gate — defense in depth at scheduler boundary; KEEP
- xj16 inline assertion structure — keep, just swap the one-line `ctx.default = ...`
- `compare_graphs` + `validate_edge_wait_coverage` — cleaner inputs, same code
- The oplb per-category count gate — **KEEP until SHADOW empirically matches CMS counts across the Phase 3 fixture surface.** v4 proposed removal under the argument "counts equal by construction." The verification reviewer correctly flagged this as premature: the construction invariant has multiple preconditions (data-flow-only categories, idMap object-identity stability, fail-loud contract catching control-op drift). Each precondition is a runtime/coding assumption that can silently break. The count gate is the only assertion that fires if/when one of those silently breaks. Removing it before all preconditions are empirically verified across the fixture surface is removing the safety net at the moment it matters most. Removal can be considered later once Phase 3 has shipped and the invariant has held across multiple kernel families.

**Acceptance:** `git grep build_non_cms_reference` returns zero hits; no test imports `pre_cms_state` or `approach_a`; the Phase 0 audit's "delete safely" list is empty.

## 5. Open questions (none remaining as blockers)

All v2 open questions are now settled:
- Q1 (reentrancy): empirically verified ✓
- Q2 (SubIter / ForceUnrollSubIter): handled inline by existing `_makeSubIterSchedule` — no extraction needed
- Q3 (TF32-emulation interleaving): handled by existing path (Approach A already proved the default scheduler emits these correctly)
- Q4 (per-body shim): not applicable — SHADOW already populates all four body slots via the existing `_captureDefaultSchedule` + `_noLoadLoopBodyDefault:3922-3935` machinery

## 6. Bead implications

| Bead | Disposition |
|---|---|
| `rocm-libraries-nmsx` (reopened) | **Critical path** — three SHADOW capture-window / scope / walk-coverage fixes (LCC, PLR1 packs, LRS/LWS pointer-math tags); IS Phase 1 of this design |
| `rocm-libraries-ltnz` | **Closed (premise factually wrong)** — proposed replacing SHADOW's "parallel walk" with idMap inversion. Direct code verification confirmed SHADOW already uses `build_idmap` + `invert_idmap_to_id_to_category`. There was no parallel walk to eliminate. |
| `rocm-libraries-6hk3` (VMovB64 classifier) | Currently OPEN. Phase 1 (nmsx) will resolve it: under the SHADOW-as-reference path, the reference contains only the CMS-subject's instructions, so VMovB64-UNKNOWN cannot surface on the reference side. Subject-side classifier registry remains globally relevant for any *future* CMS-emitted instruction class not in the registry. Close after nmsx lands and BPG#11 build proves no reference-side UNKNOWN. |
| `rocm-libraries-j4qm` (`doFullPackCodePrefetch` divergence) | Move to "real codegen-improvement opportunity, independent track" — its noise no longer affects the validator. The codegen-correctness aspect is out of scope per §1.5. |
| `rocm-libraries-oplb` (edge-layer T/X naming) | Currently CLOSED (the count gate it introduced). The closure stands. Under v5 the count gate is KEPT in Phase 4 as defense-in-depth until SHADOW empirically matches CMS counts across the Phase 3 fixture surface. The original T/X false-positive class disappears under same-codegen reference; Phase 3 verifies no T/X residuals reappear. |
| 3IJA-listed residuals (zvzu, p39d, jmfp, hcug, t4gl, gz0k, v01w) | Re-triage in Phase 3 against the hard gate. Each residual either resolves under v5 reference, becomes a real CMS-scheduler bug bead, or classifies as out-of-validator-scope per §1.5. |
| Approach A meta-beads (nyb5, y391, l1l6, hdem, m7o5) | Close as superseded; this design (and nmsx) replaces them. |

## 7. Risks

- **Each of the three nmsx fixes touches mid-build state.** Fix 1 (LCC) requires extending SHADOW's capture window past `closeLoop`, which may interact with the existing finalize-point lifetime. Fix 2 (PLR1) requires re-capturing after later subiters populate the per-iter modules, which may interact with `_capture_context.reset()` discipline at `KernelWriter.py:5723`. Fix 3 (LRS/LWS) requires inspecting which Module a leaf came from inside `_makeSubIterSchedule`, which may require restructuring the tag emission. None is independently risky, but the three together touch enough state that landing them needs careful per-fix testing.
- **Phase 3 may surface real CMS bugs.** That's a feature. Each one needs a bead. The hard gate means we can't ship Phase 4 until every residual is accounted for.
- **Empirical baseline is one fixture today.** BPG#11 alone proved SHADOW is deterministic / stable / correctly excludes j4qm noise. The risk: `build_idmap` (`ScheduleCapture.py:1042-1078`) has optional code paths (sparse-MX `LRCodeMXSA`/`LRCodeMXSB`/`LRCodeMetadata`), and the dispatcher routes differently on multiple kernel shapes (sparse / GSU / GroupedGemm / StreamK / F8 / mixed-precision). The nmsx fixes are structurally sound but only EXERCISED on whichever code paths the test fixtures cover. Mitigation: Phase 3's fixture-coverage requirement enumerates every `build_idmap` branch first, then picks fixtures that hit each.
- **Scheduler-inserted control ops** (SWaitCnt / SBarrier / SNop emitted by SIA3 itself, not inherited from source Modules) aren't in idMap. nmsx's fail-loud contract handles these via `InstructionCategory._CLASS_NAME_TO_CATEGORY` lookup with `CaptureCategoryMissingError` raise on unknown class. SIA3 adding a new control op class fails the build immediately. No silent UNKNOWN.
- **Identity tuple edge cases for unnamed LR/Pack Modules.** Covered in §3 caveat. Phase 1's tests need to explicitly exercise null `source_module_id` cases — degenerates to `canonical_render`-multiset equality, which holds under same-codegen invariant.
- **Fixture #4 may surface a new SHADOW defect class.** The third opus reviewer's prediction: nmsx's three fixes won't be the last. A future fixture may surface a fourth capture-window/scope/walk-coverage defect at a different site. Mitigation: the count-gate (kept in Phase 4) plus the fail-loud contract together catch this as a loud test failure rather than as silent drift. If a fourth defect emerges, treat it as a separate bead with its own principled justification — not as an extension of nmsx's list.

## 8. On the iteration arc

The reference design has gone through SHADOW (v1) → Approach A (m7o5 → y391 → l1l6 → hdem → nyb5) → SHADOW (v3 framing) → SHADOW + idMap-inversion (v4) → SHADOW + capture-window/scope fixes (v5). Each iteration was an exploratory pass that refined the requirements or corrected a factual error:

- v1 → v2 surfaced that synthetic re-assembly missed real instruction classes
- v2 → v3 surfaced that a separate CMS=0 build introduces codegen-branch noise that isn't the validator's concern
- v3 → v4 surfaced — incorrectly, in retrospect — that SHADOW carried a parallel categorization implementation. The proposed fix (idMap inversion) addressed a non-problem.
- v4 → v5 corrected the factual error: third-reviewer verification confirmed SHADOW already uses idMap inversion (`KernelWriter.py:4735, 4751`). The real SHADOW defects are capture-window timing (LCC), count truncation (PLR1), and one site of hand-rolled tag emission (LRS/LWS pointer-math). nmsx's original three-fix framing addresses these correctly.

This is expected — the validator sits at the intersection of CMS-scheduler choices, framework codegen, and per-arch correctness rules. Getting the reference's scope right required iterating against real fixture surfaces AND direct code verification. v4's misstep was reasoning about code at one remove rather than reading the source — a lesson worth carrying forward.

The current shape (v5) is constrained by:
- Same instruction set on both sides (precondition for asking the validator's question)
- SHADOW's capture window must cover what the production schedule emits (LCC, all subiters, per-side LRS/LWS) — Phase 1 nmsx work
- Single source of truth for categorization where applicable (already present at `:4735, 4751`; one hand-rolled site at `:1040-1045` to align)
- Fail-loud on missing categorization (no silent UNKNOWN, no per-fixture discovery)
- Scope limited to scheduler-quality (codegen correctness is a separate layer)

The constraint set is more concrete and more restrictive than v2/v3/v4. Future iterations may swap individual constraints if they prove wrong (as v4's "single source of truth as a fix" did), but the overall design surface — scheduler-quality validation against a same-instruction-set reference — is what v1-v5 have all been converging on. Back and forth is expected; the convergence is real.
