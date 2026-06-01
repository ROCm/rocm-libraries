# Round 3 verdict (post-hardening commit 0d2d79b1f8e)

## Verdict
CLEAN

## Most important finding (round 3)

The round-2 verifier's "9 unfiled SHADOW-vs-CMS divergences" finding was a **false positive caused by the cwd-import trap**, confirmed empirically. The fix agent's diagnosis is correct, and no real divergence exists. The user-facing report stating "9 pre-existing divergences" was wrong; remediation would have been wasted effort.

Direct reproduction (`bash -c 'PYTHONPATH=$WT python3 -c "import Tensile.KernelWriter; print(__file__)"'`):
- From worktree cwd: resolves to `/home/alvasile/rocm-libraries/.claude/worktrees/g9fi-impl/projects/hipblaslt/tensilelite/Tensile/KernelWriter.py` (10752 LOC, has nmsx Phase 1 Fix 1/2/3).
- From main-repo cwd: resolves to `/home/alvasile/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/KernelWriter.py` (10123 LOC, last touched by `f9abc4a545c` "rocm-libraries-p39d", which predates nmsx Phase 1).
- The 629-LOC delta and pre-nmsx state of the main-repo file is the entire mechanism producing the 9 spurious divergences.

## Per-question verification

### Q1 — cwd-trap diagnosis correct
Yes. The guard at `Tensile/Tests/unit/test_capture_pipeline_checks.py:65-82` compares `Tensile.KernelWriter.__file__`'s parent tree against `__file__`'s tree (three `..` up to the tensilelite root). Direct probe confirmed `sys.path[0]` is the cwd (empty string when running from worktree at `bash -c 'cd $WT && python3 -c "import sys; ..."'`), and cwd auto-prepend beats `PYTHONPATH` on a name clash. Main-repo KernelWriter.py at commit `f9abc4a545c` lacks nmsx Phase 1 fixes (no `_appendCloseLoopLCCToBuilder`, no per-side `LRSA/LRSB/LWSA/LWSB` tagging, no `split_for_plr` leftover walk for `LoopIters==1`).

### Q2 — count-parity test passes from worktree
Yes. Running from `/home/alvasile/rocm-libraries/.claude/worktrees/g9fi-impl/projects/hipblaslt/tensilelite` with `PYTHONPATH=$WT`:
- `test_shadow_main_capture_categories_match_cms_subject` — **PASSED**
- `test_shadow_mfma_count_matches_cms_subject_on_bpg11` (the g9fi lock-in test) — **PASSED**
- Full file: **35 passed, 1 skipped** (the skip is `test_loopiters1_per_category_count_parity` — guarded skip when the kernel-validator forces `LoopIters>=2`; not a regression). cwd-trap guard did not fire.

### Q3 — cwd-trap guard load-bearing
Yes. The guard at `:65-82` is not a silencer — it raises `RuntimeError` with a one-line directive at module load time (`:85`). Wrong-tree invocation (`bash -c 'PYTHONPATH=$WT pytest <full-worktree-path-to-test>'` from main-repo cwd) produced:
```
RuntimeError: Tensile package loaded from a different tree than this test file.
test_tree='.../worktrees/g9fi-impl/projects/hipblaslt/tensilelite',
kw_tree='/home/alvasile/rocm-libraries/projects/hipblaslt/tensilelite'.
... Fix: `cd .../worktrees/g9fi-impl/projects/hipblaslt/tensilelite` before invoking pytest.
```
The directive includes the literal `cd $WT` command needed to fix the invocation. Collection errors out with `1 error during collection`; cannot silently produce false-positive failures.

### Q4 — no new red flags
Round-3 commit (`git diff 0d2d79b1f8e^ 0d2d79b1f8e --name-only`) touches **exactly one file** (`Tensile/Tests/unit/test_capture_pipeline_checks.py`, +39 lines). No new test exclusions, no `setdefault`, no defensive classifications, no TODO without bead reference, no backwards-compat flag, no production code changes. The 39 lines are the import-time guard plus its long docstring explaining the trap and the defense.

### Q5 — broader-suite baseline
Exact match. From worktree with full unit suite (`--ignore=test_MatrixInstructionConversion.py`):
```
19 failed, 1040 passed, 4 skipped, 2 xfailed, 2 errors in 21.13s
```
Matches the per-agent expected baseline character-for-character.

### Q6 — process risk noted
Confirmed and important. The trap nearly caused dispatching remediation for non-existent problems across the nmsx/dm4p Phase 2 boundary. The hardening guard is the right durable mitigation — future verifiers will be blocked at collection time with the fix-directive in their face, not silently misled. Worth highlighting: ANY test under this project that imports `Tensile.*` and depends on worktree-specific code should adopt the same guard pattern (`_assert_tensile_tree_matches_test_tree`) — but g9fi scope does not require expanding that now.

## Recommendation

**CLEAN.** Dispatch the correctness verifier (Step 6 / task #297). The two commits on `g9fi-impl` (`a9576196e6e` original + `0d2d79b1f8e` hardening guard) constitute a complete and accurate fix:
- The MFMA exclusion removal is correct (test_shadow_mfma_count_matches_cms_subject_on_bpg11 PASSES with both sides at 48).
- The count-parity test passes when invoked correctly.
- The 9 "divergences" reported in round 2 do not exist; they were an artifact of pytest cwd-import shadowing.
- The hardening guard prevents the false-positive class from recurring.

No new beads required. No production behavior changed.

---

# g9fi Plan-Adherence Verification (Round 2 — superseded)

## Verdict
NEEDS-FIXES

## Most important finding

The agent's central technical claim is **correct**: the g9fi bead's premise was empirically wrong. CMS's `customMainLoopSchedule` flattens `MfmaCodeAllIters` via `removeComments(...)` at `Tensile/Components/CustomSchedule/dispatch.py:123` BEFORE the tagging loop runs, and `removeComments` itself calls `module.flatitems()` at `dispatch.py:75-78` (returns a flat list of leaves). The subsequent dispatch loop at `dispatch.py:235-240` then tags each leaf as MFMA (`tag_by_origin_id[id(mfmaItem)] = "MFMA"`, `macro.add(mfmaItem)`). Empirically on BPG#11 CMS emits 48 per-leaf MFMA tags — not 1 per Module — and SHADOW also emits 48; the lock-in test `test_shadow_mfma_count_matches_cms_subject_on_bpg11` PASSES. The atomic-Module mechanism the bead prescribed would have collapsed SHADOW's 48 down to 2 while CMS continued to produce 48 — i.e. introduced divergence. Removing MFMA from the count-parity exclusion is therefore the correct call. **However**, the agent's commit message contains a FALSE verification claim: it states `test_shadow_main_capture_categories_match_cms_subject` PASSES without the MFMA exclusion. It does NOT pass — there are 9 OTHER pre-existing divergent categories (LRS, LWS, LCC, PackA3, PackB3, LRSA, LRSB, LWSA, LWSB). These divergences existed at validator tip 51636ca2eb2 (baseline), so g9fi does not introduce them — but per the standing rule they should be filed as a P0 blocker bead because they sit between this fix and rocm-libraries-dm4p (Phase 2 wire SHADOW as ctx.default), and the test now fails as a non-MFMA-related red flag.

## Per-question verification

### Q1 — Bead premise wrong?
**VERDICT: Bead premise is empirically wrong; agent's resolution is correct.**

Evidence:
- `Tensile/Components/CustomSchedule/dispatch.py:73-78`: `removeComments(module)` constructs `retModule = Module()`, iterates `module.flatitems()`, filters TextBlock/SCBranchSCC1/SNop, and returns `retModule.flatitems()` — a flat list of leaf instructions.
- `Tensile/Components/CustomSchedule/dispatch.py:123`: `mfmaCode = removeComments(mfmaCode)` — the MfmaCodeAllIters Module is replaced with a flat leaf list BEFORE any dispatch-loop iteration.
- `Tensile/Components/CustomSchedule/dispatch.py:235-240`: `for miIndex in range(-1, len(mfmaCode)): ... mfmaItem = mfmaCode[miIndex]; tag_by_origin_id[id(mfmaItem)] = "MFMA"; macro.add(mfmaItem)`. Each leaf in the flat list is individually tagged. The mfmaIter Module wrapper is gone by this point.
- Test `test_shadow_mfma_count_matches_cms_subject_on_bpg11` (lines 888-952) executes against the real BPG#11 build and confirms both sides produce 48 MFMA tags.
- Direct empirical confirmation: the count-parity test output at HEAD shows `SHADOW={'MFMA': 48}, CMS={'MFMA': 48}` — MFMA is absent from the mismatch list.

### Q2 — Lock-in test load-bearing?
**VERDICT: Yes, load-bearing and real.**

Evidence:
- `Tensile/Tests/unit/test_capture_pipeline_checks.py:920` calls `_build_bpg11_writer_and_capture()` which at line 596/620 uses `cms_test_utils._make_solution` to build a real BPG#11 kernel through `KernelWriterAssembly._getKernelSource(solution)`. Not a mock.
- The test extracts SHADOW main_loop via `_shadow_main_body(writer)` (line 652) which reads `writer._test_shadow_capture` from a monkey-patched `build_non_cms_reference` snapshot, and CMS main_loop via `_cms_main_body(writer)` (line 669) from `writer._capture_context.cms.main_loop`. Both come from the real capture pipeline.
- Asserts three things (lines 935-952): SHADOW MFMA == 48, CMS MFMA == 48, and SHADOW == CMS. A regression in either dispatch.py's per-leaf tag loop OR KernelWriter.py's per-leaf MFMA walk would fail at least one assertion.
- Verified PASSING under `pytest -v -k "shadow_mfma_count"`.
- Test would have caught the bead's prescribed fix: if SHADOW were converted to atomic-Module treatment, SHADOW would drop to 2 and `shadow_mfma == 48` would fail.

### Q3 — setdefault → direct assignment justified?
**VERDICT: Justified; semantically cosmetic-but-honest.**

Evidence:
- `Tensile/KernelWriter.py:4981`: `macIterCode.add(deepcopy(mfmaIter))`. The deepcopy clones the entire mfmaIter Module subtree, producing fresh Python `id()` values for every interior leaf.
- `Tensile/KernelWriter.py:5224-5225`: `for _leaf in macIterCode.flatitems(): capture_id_to_cat[id(_leaf)] = "MFMA"`. Since each `id(_leaf)` is a freshly-allocated post-deepcopy id, there cannot be a prior entry in `capture_id_to_cat` (which is built from the source modules — different objects). Direct assignment and `setdefault` produce identical end-states.
- The change correctly removes a red-flag silent-fallback pattern (`.setdefault` is anti-pattern under the standing rule).

### Q4 — Per-leaf walk legitimately kept?
**VERDICT: Yes, legitimately kept for fail-loud coverage.**

Evidence:
- `Tensile/KernelWriterAssembly.py:7624`: `def mfmaIter(...)` starts with `imod = Module("mi")` and returns `imod`. Everything added to `imod` becomes part of the mfmaIter Module that the deepcopy at KernelWriter.py:4981 clones.
- `Tensile/KernelWriterAssembly.py:8367`: `imod.add(SNop(waitState=(s_nop - 1), comment=""))` is emitted inside mfmaIter when `s_nop != 0` (line 8364-8366 path). SNop leaves would otherwise have no SHADOW idmap entry and trigger CaptureCategoryMissingError.
- `Tensile/KernelWriterAssembly.py:8622`: `imod.add(SWaitAlu(va_vdst=0, ...))` is emitted inside mfmaIter when `kernel["ExpertSchedulingMode"] > 0`. Same fail-loud trigger if walk removed.
- `Tensile/KernelWriterAssembly.py:8362`: `shiftK.add(VCndMaskB32(...))` is added inside the shiftK control region of mfmaIter — emits non-MFMAInstruction leaves that would also trip fail-loud without the walk.
- CMS-side counterpart: `removeComments` filters SNop (line 76: `not isinstance(i, (SCBranchSCC1, SNop))`) so SNop never reaches the CMS tag map. The SHADOW walk's "tag everything MFMA" gives those leaves a category so fail-loud doesn't fire, mirroring CMS's "they don't exist" behavior at the slot-kind boundary.

### Q5 — No backwards-compat / hacks / punted work?
**VERDICT: NEEDS-FIXES — punted work not filed as P0.**

Evidence:
- The diff contains zero new `setdefault` calls, no new feature flags, no new kwargs, no new TODO/for-now comments. The agent removed one setdefault (KernelWriter.py:5225) and made it direct assignment. No new test exclusions added; one exclusion (MFMA) removed.
- **HOWEVER**: the agent removed MFMA from the count-parity exclusion list, and the resulting test now FAILS on 9 pre-existing non-MFMA divergences: `[('LRS', 2, 0), ('LWSA', 0, 1), ('LRSB', 0, 1), ('PackA3', 0, 20), ('LWS', 2, 0), ('LWSB', 0, 1), ('LRSA', 0, 1), ('LCC', 0, 2), ('PackB3', 0, 20)]`. The agent's commit message claims this test PASSES — that claim is empirically false (see Q6). These divergences existed at validator tip 51636ca2eb2 too (baseline run via `/tmp/baseline_test_capture_pipeline_checks.py` confirms identical 9-element mismatch list), so they are not introduced by g9fi. But per the standing rule ("If you discover new work mid-implementation, file it as a P0 blocker on the next dependent bead"), these divergences are blockers for `rocm-libraries-dm4p` (Phase 2: wire SHADOW as ctx.default) — that bead cannot be safely landed while SHADOW and CMS disagree on LRS/LWS/LCC/PackA3/PackB3/LRS{A,B}/LWS{A,B} counts. The agent did not file this.

### Q6 — Tests pass?
**VERDICT: NEEDS-FIXES — agent's claim about count-parity test is false, but no regression introduced.**

Evidence (run from `/home/alvasile/rocm-libraries/.claude/worktrees/g9fi-impl/projects/hipblaslt/tensilelite`, pytest `-P` flag rejected as unrecognized by pytest 9.0.3 in `/home/alvasile/venv`; ran without it):

1. `test_capture_pipeline_checks.py` at HEAD: **10 failed, 25 passed, 1 skipped**. The new lock-in test PASSES. The count-parity test (`test_shadow_main_capture_categories_match_cms_subject`) FAILS with 9 non-MFMA mismatches:
   ```
   AssertionError: Per-category count mismatches on non-mfmaIter-sub-leaf data-flow categories:
   [('LRS', 2, 0), ('LWSA', 0, 1), ('LRSB', 0, 1), ('PackA3', 0, 20), ('LWS', 2, 0),
    ('LWSB', 0, 1), ('LRSA', 0, 1), ('LCC', 0, 2), ('PackB3', 0, 20)]
   SHADOW={'MFMA': 48, ...}, CMS={'MFMA': 48, ...}
   ```

2. Baseline at validator tip 51636ca2eb2 (test file checked out to `/tmp/baseline_test_capture_pipeline_checks.py`): **10 failed, 24 passed, 1 skipped**. Same count-parity test fails with the SAME 9 non-MFMA mismatches.

3. Full unit suite at HEAD: **29 failed, 1030 passed, 4 skipped, 2 xfailed, 2 errors**. The user's expected baseline of `19 failed, 1040 passed, 4 skipped, 2 xfailed, 2 errors` does NOT match. The 10 extra failures all live in `test_capture_pipeline_checks.py` and pre-date g9fi (baseline test file confirms). The user's expected-baseline numbers in the verification instructions appear to be incorrect (possibly state-drifted), but the more important fact is that g9fi introduces no NEW failures vs validator tip — the 10 `test_capture_pipeline_checks.py` failures and the broader unit failures (test_ScheduleCapture.py, test_cms_from_default.py, test_dataflow_graph_emission_ordinal.py, test_prologue_capture.py) are all pre-existing.

4. MFMA equality is empirically confirmed (in the failure output above, MFMA does NOT appear in the mismatch list and both sides report 48). The agent's central claim is sound.

### Q7 — Process observation
The bead `rocm-libraries-g9fi` was filed based on a model of CMS-side MFMA tagging that did not survive empirical inspection. The bead description at `br show rocm-libraries-g9fi` claims "CMS-side `expand_cms_macro` at `ScheduleCapture.py:2443` and `dispatch.py:240` treats mfmaIter as a single MFMA-tagged Module" — but `dispatch.py:235-240` iterates the flattened leaf list, not the Module wrapper. The round-2 plan-adherence verifier accepted this framing without verifying the CMS-side per-Module claim against the actual `removeComments` flattening at `dispatch.py:73-78` / `:123`. This is a worth-noting process observation: future verifiers should always run an empirical probe before accepting "structural divergence" framings.

## Concerns

### Concern 1: Agent's commit message contains a false verification claim
**Severity:** MEDIUM
**Evidence:** Commit `a9576196e6e` message states "`test_shadow_main_capture_categories_match_cms_subject` PASSES without the MFMA exclusion." Empirically the test FAILS post-commit (and also fails at baseline) with 9 non-MFMA mismatches. The agent likely did not actually re-run the test after removing MFMA from the exclusion, or interpreted "MFMA-related portion passes" as "test passes."
**Recommended fix:** Update the commit message (or have the agent acknowledge) that the test currently still fails due to PRE-EXISTING non-MFMA divergences, and that those divergences are not introduced by g9fi. This is honest framing.

### Concern 2: 9 pre-existing non-MFMA count-parity divergences NOT filed as P0 blocker bead
**Severity:** BLOCKING
**Evidence:** Removing MFMA from the count-parity exclusion exposes a test that fails on `[LRS, LWSA, LRSB, PackA3, LWS, LWSB, LRSA, LCC, PackB3]`. The standing rule says: "If you discover new work mid-implementation, file it as a P0 blocker on the next dependent bead." The next dependent bead is `rocm-libraries-dm4p` (Phase 2 wire SHADOW as ctx.default, P0, currently open and listed as blocked by g9fi). Wiring SHADOW as the default reference while it disagrees with CMS on LRS/LWS/LCC/PackA3/PackB3/LRSA/LRSB/LWSA/LWSB counts will silently produce wrong results. Yet g9fi is currently slated to "resolve" — which will unblock dm4p — without these divergences being tracked.
**Recommended fix:** Before declaring g9fi resolved, file a new P0 bead (e.g. `rocm-libraries-<new-id>: SHADOW vs CMS count-parity divergences on LRS/LWS/LCC/PackA3/PackB3/LRSA/LRSB/LWSA/LWSB`) and `br dep add` it as a blocker on `rocm-libraries-dm4p`. The new bead should document the exact pre-existing 9-element mismatch list captured at validator tip 51636ca2eb2 and post-g9fi commit a9576196e6e (both are identical), and target a real root-cause fix rather than re-adding any test exclusion.

### Concern 3: User's stated expected baseline numbers don't match observed reality
**Severity:** LOW (informational — does not block g9fi)
**Evidence:** User's instructions expect `19 failed, 1040 passed, 4 skipped, 2 xfailed, 2 errors`. Observed at HEAD: `29 failed, 1030 passed, 4 skipped, 2 xfailed, 2 errors`. The 10-failure delta lives entirely in `test_capture_pipeline_checks.py` and pre-dates g9fi (confirmed by running baseline file). The user's expectation appears state-drifted.
**Recommended fix:** None for g9fi itself. The validator's baseline-expected-results spec should be refreshed against the current validator tip.

## Recommendation

**NEEDS-FIXES** before declaring g9fi resolved.

Required actions in order:

1. Agent (or verifier on their behalf) files a new P0 bead for the 9 pre-existing non-MFMA count-parity divergences, with `br dep add` linking it as a blocker on `rocm-libraries-dm4p`. The bead description should reference: `Tensile/Tests/unit/test_capture_pipeline_checks.py::test_shadow_main_capture_categories_match_cms_subject` and document that the mismatch list `[('LRS', 2, 0), ('LWSA', 0, 1), ('LRSB', 0, 1), ('PackA3', 0, 20), ('LWS', 2, 0), ('LWSB', 0, 1), ('LRSA', 0, 1), ('LCC', 0, 2), ('PackB3', 0, 20)]` is identical pre- and post-g9fi.

2. Honest acknowledgment in commit message (amend or follow-up) that `test_shadow_main_capture_categories_match_cms_subject` does NOT currently pass — only the MFMA portion of its parity check passes; the test still fails on pre-existing non-MFMA divergences tracked in the new bead.

3. Once (1) and (2) are done, the g9fi resolution itself is principled and correct. Dispatch correctness verifier (Step 6) can then proceed.

If the user explicitly decides that pre-existing divergences are acceptable to leave un-tracked (overriding the standing rule), then the verdict downgrades to CLEAN — but my recommendation is to file the blocker first.
