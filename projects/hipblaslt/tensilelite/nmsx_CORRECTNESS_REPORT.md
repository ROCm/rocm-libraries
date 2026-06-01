# Round 2 verdict (post-fix commit e6cdc49dae9)

## Verdict
**CLEAN** — both latent correctness bugs identified in round 1 are RESOLVED. The fix is structurally sound, mirrors CMS's `dispatch.py` semantics correctly, introduces no new hacks/workarounds, and the broader test baseline is preserved exactly (`19 failed, 1039 passed, 4 skipped, 2 xfailed, 2 errors`).

## Per-bug-fix status

### Bug 1 (was HIGH): RESOLVED
**Evidence:**
- New filter at `KernelWriter.py:2975-2987` (`_appendCloseLoopLCCToBuilder`) captures every `Instruction` leaf, skipping only `SCBranchSCC1` + `SNop`. Verified via `/home/alvasile/venv/bin/python3` runtime check: `SCmpEQU32` and `SCSelectB32` are both `Instruction` subclasses (`SSubU32.__mro__` shows `CommonInstruction > Instruction > Item > object`), so the PGR=2 4-leaf path is now fully captured.
- The skip set MATCHES `dispatch.py:73-78` (`removeComments`): dispatch.py skips `TextBlock + (SCBranchSCC1, SNop)`; the fix skips `not isinstance(item, Instruction)` (covers TextBlock, Label, and any other non-Instruction Item) + `(SCBranchSCC1, SNop)`. The non-Instruction skip is BROADER than `removeComments` — but legitimately so, because the CMS path (`finalLoop=False`) never has Labels in `loopCounterCode`, and the Approach-A path (`finalLoop=True`) DOES emit Labels (`KernelWriterAssembly.py:6907-6913`) which `LoopBodyCaptureBuilder` cannot wrap (Labels lack `reads_scc`/`writes_scc`; `_SCCRule.applies` at `ScheduleCapture.py:1854` would AttributeError).
- The Label-skip is NOT a hidden carve-out: Labels are control markers, never data-flow producers, never categorized as LCC by CMS (CMS doesn't see them post-removeComments-on-`finalLoop=False`-output).
- Unit tests (`TestAppendCloseLoopLCCToBuilderPgr2`, 4 tests) PASS: PGR=2 4-leaf capture, TextBlock/SCBranchSCC1/SNop skip, Label skip on Approach-A path, default 2-leaf regression guard.
- `test_approach_a_non_cms_reference.py` (4 tests) PASS — confirms the broadened filter doesn't break the Approach-A consumer; the Instruction-only restriction handles Labels correctly.

### Bug 2 (was MEDIUM): RESOLVED (focused-unit-level; end-to-end skip is legitimate)
**Evidence:**
- New code at `KernelWriter.py:5339-5349` mirrors `dispatch.py:103-111` exactly: when `len(LRCodeAAllIters) == 1` (which equals `kernel["LoopIters"]`), applies `split_for_plr` to each LRCode[0]/PackCode[0] and rebuilds `idmap` with `num_loop_iter=2`. The gate condition matches dispatch.py byte-for-byte.
- `split_for_plr` at `ScheduleCapture.py:966-980` returns `[items[n//2:], items[:n//2]]` — pure Python list slicing on the result of `module.flatitems()`. Items are NOT cloned. The agent's id-preservation claim is verified by `test_split_preserves_leaf_identity_for_cross_side_lookup` which directly asserts `combined_ids == original_ids`.
- For `LoopIters >= 2`, the new code path is NOT triggered (`if _num_loop_iter_for_idmap == 1:` gate); behavior unchanged. Verified by `test_no_split_for_plr_on_loopiters_1_misses_pack_a1_pack_b1` which exercises the unsplit path.
- Unit tests (`TestLeftoverIdmapSplitForPlrLoopIters1`, 3 tests) PASS: post-split idmap shape (PackA0/A1, PackB0/B1, LRA0/A1, LRB0/B1), pre-fix regression guard, identity preservation.

## Round-2 concerns

### NCR1 (Approach-A test breakage history): RESOLVED
The agent's reported intermediate-state failure on `test_approach_a_non_cms_reference.py` was caused by the broadened capture sending Labels through `LoopBodyCaptureBuilder` → `_populate_wrapper` → `_SCCRule.applies` (which does `inst.reads_scc or inst.writes_scc`). Labels lack those C++-bound fields. The correct fix was to restrict the filter (as done), NOT to teach the receiver to handle Labels — Labels are not data-flow producers, not LCC, and have no semantic role in the captured body. CMS already implicitly relies on this via dispatch.py's `closeLoop(finalLoop=False)` call that emits no Labels.

### NCR2 (new hacks/workarounds): NONE INTRODUCED
Diff review (`git show e6cdc49dae9`) shows no new `setdefault`, `# TODO`, `# FIXME`, `# HACK`, `# for now`, `defensive` (in a hack sense), or test-exclusion patterns. The token "skip" only appears in: (a) docstrings/comments describing what the filter skips (TextBlock, Label, branches), (b) test names verifying skip behavior (`test_skips_label_on_approach_a_finalloop_path`), (c) `pytest.skip` on legitimately unbuildable fixtures.

### NCR3 (baseline match): VERIFIED
Ran broader suite myself (with `-P` python flag + `--ignore=test_MatrixInstructionConversion.py`):
```
19 failed, 1039 passed, 4 skipped, 2 xfailed, 2 errors in 21.01s
```
Failure classification matches round-1 baseline exactly — all in Approach-A path (`test_ScheduleCapture.py`, `test_cms_from_default.py`, `test_dataflow_graph_emission_ordinal.py`, `test_prologue_capture.py`, `test_cross_subiter_alu_carveout_real_kernel.py`). No new regressions.

### Corner cases investigated (now that PGR=2 + LoopIters==1 are addressed)
- **ExpandPointerSwap / DirectToVgpr**: Only affect `closeLoop` inside the `if finalLoop:` block (`KernelWriterAssembly.py:6903-6963`). CMS calls `closeLoop(finalLoop=False)` — no impact on SHADOW LCC capture. SAFE.
- **WaveSeparateGlobalRead / 1LDSBuffer=0**: Affect per-iter LR/LW structure routed through `_makeSubIterSchedule`'s per-side leaf-id capture (round-1 Fix 3). No new edge cases at the closeLoop/leftover-walk surface. SAFE.
- **Mixed-precision**: same source-module structure — capture pipeline is class-agnostic. SAFE.
- **OptNoLoadLoop**: Already explicitly rejected with `NotImplementedError` at `KernelWriter.py:5257-5264`. SAFE.
- **Sparse-MX**: Already fail-loud via `CaptureCategoryMissingError`; intentional Phase 3 discovery mechanism. SAFE.

No new P0-blocking corner-case fixtures discovered.

## Test surface
- Baseline match: VERIFIED (19 failed + 2 errors, all Approach-A-path, matches commit `f6891aab835` tip).
- New focused tests: 7 new tests in `TestAppendCloseLoopLCCToBuilderPgr2` (4) + `TestLeftoverIdmapSplitForPlrLoopIters1` (3) all PASS.
- LoopIters==1 end-to-end skip: LEGITIMATE — the BPG#11 base shape (DepthU=32, MI K=32) hits "reject: UseCustomMainLoopSchedule=1 but CMS is not supported" inside `_make_solution`, returning None from the fixture builder. The CMS schedule registry has no entry for LoopIters==1 with the current BPG#11-style configs. The focused unit tests (`TestLeftoverIdmapSplitForPlrLoopIters1`) directly verify the load-bearing contracts: post-split idmap shape, regression guard against unsplit shape, and `split_for_plr` identity preservation. End-to-end coverage will land in Phase 3 when a CMS-registered LoopIters==1 schedule is added.
- The pytest.skip message says "amdclang/clang not available" but the real cause is upstream `_make_solution` exception; this is a minor diagnostic-message bug in the test, NOT a correctness defect in the implementation. Could be fixed by surfacing the underlying exception, but not blocking.

## New beads filed
NONE — no new defects discovered that warrant P0 blockers. The 3 existing round-1 follow-up beads remain valid:
- Bug 1 + Bug 2 latent-bug beads are now SUPERSEDED by the fixes in this commit.
- MFMA-shape follow-up bead (`rocm-libraries-g9fi`) referenced in the updated MFMA-exclusion comment is the remaining documented punt for Phase 3.

## Recommendation
**CLEAN — ready for Step 9 (squash-merge).**

Both latent bugs are resolved correctly via direct mirroring of `dispatch.py` semantics. The Label-skip is legitimate (semantic, not carve-out). No new hacks. Baseline preserved. Focused unit tests cover all load-bearing contracts. The end-to-end LoopIters==1 test skip is legitimate fixture coverage absence (no registered CMS schedule for that shape) and is properly documented + flagged by the test logic itself (lines 1469-1484 explicitly skip rather than silently pass when LoopIters != 1).

---

# Round 1 report (preserved for history)

# nmsx Correctness & Soundness Report

## Verdict
**HAS-CORRECTNESS-CONCERNS** — code is functionally correct for the BPG#11-shaped fixture currently exercised. Two latent correctness issues identified that will manifest on legitimate but un-exercised kernel shapes (PGR=2 with ASEM%64==0; LoopIters==1 with CMS). Neither blocks the BPG#11 use-case; both are fixture-coverage gaps that Phase 3 must address before declaring nmsx complete.

## Per-fix correctness

### Fix 1 — LCC absence
**Verdict: CORRECT in BPG#11 path; LATENT BUG on PGR=2 + AssertSummationElementMultiple%(DepthU*2)==0 path.**

**Verified:**
- `closeLoopMod` is computed once at `KernelWriter.py:5267` and the SAME instance is fed to both:
  - `build_idmap(loopCounterCode=closeLoopMod, ...)` at `:5304`
  - `customMainLoopSchedule(..., closeLoopMod if ...)` at `:5400`
  This is the same Python Module instance, so `removeComments(closeLoopMod)` inside `dispatch.py:122` walks the same physical `SSubU32`/`SCmpEQI32` instances — leaf-id identity is preserved, idMap inversion works.
- `_appendCloseLoopLCCToBuilder` at `:2907-2955` correctly filters for `SSubU32` + `SCmpEQI32` via isinstance.
- `closeLoop` is only called ONCE per `_loopBody` invocation in the CMS path (the `else self.closeLoop(...)` fallback at `:5400` is dead code under the outer `if kernel["UseCustomMainLoopSchedule"]:` gate at `:5206` — `closeLoopMod` is always set at `:5266-5267`).
- Empirical: BPG#11-shaped test fixture (`test_capture_pipeline_checks.py:_build_bpg11_writer_and_capture`) shows `LCC=2` on both SHADOW and CMS sides (verified by direct probe).
- `customMainLoopSchedule` does not mutate the `loopCounterCode` Module in place — `removeComments` returns a new list of items.

**Latent bug — PGR=2 LCC undercount in SHADOW:**
- `KernelWriterAssembly.py:closeLoop` at `:6845-6892` has a conditional branch (`if kernel["PrefetchGlobalRead"]==2:`) at `:6845`, GATED on `kernel["AssertSummationElementMultiple"] % (DepthU * 2) == 0` at `:6837-6838`.
- When both conditions hold, closeLoopMod contains 4 leaves: `SCmpEQU32 + SCSelectB32 + SSubU32 + SCmpEQI32` (lines `:6848, :6849, :6850, :6857`).
- `_appendCloseLoopLCCToBuilder:2951` only matches `(SSubU32, SCmpEQI32)`. The two additional control leaves (`SCmpEQU32`, `SCSelectB32`) are silently skipped.
- On the CMS side, `customMainLoopSchedule` -> `build_idmap(loopCounterCode=closeLoopMod, ...)` tags ALL four leaves as "LCC" via `invert_idmap_to_id_to_category`, and `emit_instructions` at `dispatch.py:305-314` emits all four.
- Net: in PGR=2 + ASEM%(DepthU*2)==0 kernels, SHADOW LCC=2 but CMS LCC=4. The parity test at `test_capture_pipeline_checks.py:856` does NOT exclude LCC, so a fixture exercising this would FAIL `test_shadow_main_capture_categories_match_cms_subject`.
- Test fixture (`AssertSummationElementMultiple` default = 1; not a multiple of 64) currently does NOT trigger this.
- **Severity: HIGH** — design v5 §4 Phase 3 fixture-coverage requirement names PGR variants explicitly; closing this requires either (a) extending `_appendCloseLoopLCCToBuilder` to include `SCmpEQU32` + `SCSelectB32`, or (b) restructuring to harvest every leaf of `closeLoopMod` (filtering by category from build_idmap inversion).

**Latent issue — closeLoop side-effects relocated:**
- `closeLoop` mutates `self.states.ldsWriteTokenIdx` at `KernelWriterAssembly.py:6967`, checks-in `self.oriLraA/B/M` and `self.oriLwaA/B/M` at lines ~7007-7053, sets them to None.
- The fix moves the `closeLoop()` call from inside `customMainLoopSchedule` (formerly at `_loopBody:5400` execution time) to ~135 lines earlier at `_loopBody:5267`. The intervening code (5275-5394) does not read `ldsWriteTokenIdx`, `oriLraA/B`, or `oriLwaA/B`, so no observable downstream impact. Round-1 verifier's "no observable side effects" claim was reductive (there ARE side effects) but the net empirical conclusion stands.

### Fix 2 — PLR1 packs
**Verdict: CORRECT for LoopIters>=2; LATENT BUG for LoopIters==1.**

**Verified:**
- `len(LRCodeAAllIters)` at `_loopBody:5290` equals `kernel["LoopIters"]` because `LRCodeAAllIters.append(Module())` runs unconditionally for every `uIdx in range(0, kernel["LoopIters"])` at `:4485` (gated only on `UseCustomMainLoopSchedule`, which is always True in this branch).
- Pre-skip via `builder._instructions` at `KernelWriter.py:5326` correctly accesses the private list; this is fragile (no public accessor at `LoopBodyCaptureBuilder`) but works.
- Per-iter prefetch_pack walks at `:5312-5317` mirror the per-iter tagging at `:5026-5031`.
- Empirical: SHADOW PackA0/PackB0/PackA3/PackB3 = 20/20/20/20 ; CMS = 20/20/20/20. Per-category count parity verified.
- The xbi0 (same-id) and flpk (canonical-text cross-tagging) invariants are pre-checked at `:5348` and `:5374-5376` respectively before each `builder.append`.
- The `setdefault` walk for prefetch_pack at `:5026-5031` (cited in task as "around :4763-4768") is in the per-uIdx loop body and runs every uIdx; this is correct.

**Latent bug — LoopIters==1 / split_for_plr divergence:**
- `customMainLoopSchedule:dispatch.py:103` calls `scap.split_for_plr(LRCodeA[0])` when `numLoopIter == 1`, producing 2 buckets. The CMS-side `build_idmap` then runs with `num_loop_iter=2` and tags pack leaves as `PackA0/PackA1/PackB0/PackB1`.
- The SHADOW leftover walk at `KernelWriter.py:5290` uses `num_loop_iter=len(LRCodeAAllIters) = 1` (raw `kernel["LoopIters"]`), tagging only `PackA0/PackB0`. No `split_for_plr` is applied.
- Result: in LoopIters==1 kernels, SHADOW will have N pack leaves under `PackA0` while CMS has N/2 under `PackA0` + N/2 under `PackA1`. **Count parity test would fail.**
- BPG#11 has LoopIters=2 so this path is not exercised.
- **Severity: MEDIUM** — design v5 §3 ("identity uses `(canonical_render, source_module_id)` tuple-set equality") suggests this case may need special handling. The fix would mirror `split_for_plr` on the SHADOW side when `LoopIters==1`.

### Fix 3 — LRS/LWS schema
**Verdict: CORRECT and complete for the production scope.**

**Verified — production call sites for `localReadSwapOffsets`/`localWriteSwapOffsets`/`localReadInitPointers`:**
- Inside `_loopBody` (mainloop SHADOW scope): `:4782-4906` — all are wrapped with per-side leaf-id capture into `pointer_lr*/lw*_leaf_ids_{A,B}` sets. PASS.
- Inside `_noLoadLoopBodyDefault` (NLL/NGL SHADOW scope): `:3777-3908` — same wrapping pattern. PASS.
- Outside these two scopes (the SHADOW capture's main_loop and n_gl/n_ll bodies): `:3285-3296`, `:5627-5728`, `:6000-6009`, `:6790-6804`. These emit into pre-loop / kernel-body / setupNewTile context — they do NOT flow through `_makeSubIterSchedule`'s pointer walk, so they cannot reach the SHADOW capture pipeline. The non-coverage here is correct.

**Verified — `_makeSubIterSchedule` kwarg:**
- New kwargs `capture_pointer_side_map={...}`, `capture_body_label="main_loop"`, `capture_fail_loud_on_missing_category=True` at `:986-992`. Defaults preserved.
- When `capture_pointer_side_map` is None (`:1045-1046`), defaults to `{}` and per-side sets default to `frozenset()` (`:1047-1050`). Leaves get no LRSA/LRSB/LWSA/LWSB tag — they fall through to `_captureSubIterToBuilder` which, with `fail_loud_on_missing_category=True`, would raise `CaptureCategoryMissingError`. So the default behavior on Approach-A-style callers that don't set the kwarg (and don't opt out of fail-loud) would fail.
- All production CMS+SHADOW callers pass `capture_pointer_side_map`. Verified at `:4014-4020` (NLL/NGL SHADOW), `:5043-5048` (Approach-A non-CMS capture site — which ALSO opts out of fail-loud at `:5060`), `:5186-5191` (mainloop SHADOW).

**Sparse-MX metadata handling:**
- Per round-2 fix, metadata leaves are NO LONGER pre-classified as LRSB. They are emitted into `pointerLRCode` but not recorded in any per-side set. With `fail_loud=True`, sparse-MX fixtures would trigger `CaptureCategoryMissingError`. The comment at `:4854-4863` explicitly names this as intentional: "Per design v5 §4 Phase 1 fail-loud contract: surface un-bucketed leaves via CaptureCategoryMissingError instead of defaulting them."
- No active sparse-MX unit test fixtures exist (`grep -l "Sparse" Tensile/Tests/unit/*.py` shows 2 files, both unrelated to schedule capture). The fail-loud will not fire on current test surface but WILL fire when Phase 3 introduces a sparse-MX fixture — that's the intended design.

### Fail-loud contract
**Verdict: CORRECT.**

- `CaptureCategoryMissingError` defined at `ScheduleCapture.py:117-138` with the required context (rocisa class, body-label, subiter, leaf-repr, design citation).
- Raise site at `KernelWriter.py:2838-2855` includes: class name, idMap-lookup-failed signal, registry inst_cat name, body_label, subiter, leaf-repr, design-doc citation. Sufficient for debugging from the message alone.
- Single explicit-False audit (`grep -rn fail_loud_on_missing_category=False Tensile/`):
  - `KernelWriter.py:5060` — Approach-A `_captureNonCmsBuild` call site. Has Phase 4 TODO at `:5049-5059` tying its lifetime to Approach-A deletion. CORRECT.
  - `test_capture_pipeline_checks.py:982` — canary test that exercises the silent path. CORRECT.
- Function-level defaults: `_makeSubIterSchedule` at `:989` -> True; `_captureSubIterToBuilder` at `:2712` -> True. CORRECT (principled-default, legacy-opt-out).
- The silent UNKNOWN path at `:2872` is reachable only via explicit-False. After Phase 4 deletes the Approach-A call site, this else arm becomes dead code (TODO at `:2868-2871`). CORRECT.

### Tests
**Verdict: CORRECT — tests genuinely exercise the contracts they claim.**

- 27 tests; all pass. Pre-existing 19 failures + 2 errors elsewhere reproduce baseline.
- The 7 new tests in `TestShadowCaptureNmsxFixes` + `TestShadowCaptureFailLoudOnUnknownCategory`:
  - `test_shadow_main_capture_contains_lcc` (`:696-725`): asserts `len(lcc_tagged) >= 1` AND that the leaf classes include `SSubU32` or `SCmpEQI32`. Tests the contract.
  - `test_shadow_main_capture_contains_per_subiter_packs` (`:727-786`): per round-2, now asserts per-(category) COUNT equality AT `:773-786`, not just set membership. CORRECT — addresses round-1 Concern 3.
  - `test_shadow_main_capture_uses_per_side_lrs_lws_tags` (`:788-809`): verifies absence of unsided "LRS"/"LWS". Tests the contract.
  - `test_shadow_main_capture_categories_match_cms_subject` (`:811-882`): broad per-category parity excluding `{SYNC, SNOP, SSETPRIO, SBARRIER, MFMA}`. Exclusion list is minimal-and-justified.
  - `test_synthetic_unregistered_class_raises` (`:908-954`): uses VXorB32 (verified absent from `_CLASS_NAME_TO_CATEGORY` via in-test precondition assertion); calls the real `_captureSubIterToBuilder` (not a mock) via `SimpleNamespace` shim; asserts message contains "VXorB32", "main_loop", "DEFAULT_SCHEDULER_REFERENCE_DESIGN". Sound.
  - `test_synthetic_unregistered_class_silent_when_fail_loud_off` (`:956-985`): asserts silent UNKNOWN path works. Sound.
- The BPG#11 fixture (`_build_bpg11_writer_and_capture:565-649`) uses the real `_make_solution` + `KernelWriterAssembly` path, not a mock. It monkey-patches `build_non_cms_reference` to snapshot SHADOW before xj16 overwrites it — a legitimate test instrumentation, not test-bypassing.

**MFMA exclusion verified:**
- `excluded = {"SYNC", "SNOP", "SSETPRIO", "SBARRIER", "MFMA"}` at `test_capture_pipeline_checks.py:856`. NO other categories silently absent. Confirmed by reading the parity test in full.
- The MFMA exclusion is documented at `:819-841` with the per-leaf-vs-per-Module structural reason; the round-2 verifier already flagged the lack of follow-up bead as the remaining open item.

## Bugs found

### Bug 1: PGR=2 + ASEM%(DepthU*2)==0 produces SHADOW LCC undercount
**Severity:** HIGH (latent — does not fire on current test surface)
**File:line:** `KernelWriter.py:2951` (the `(SSubU32, SCmpEQI32)` isinstance filter); root cause is `KernelWriterAssembly.py:6845-6859` (PGR=2 emits 4 LCC-class leaves).
**Description:** In the PGR=2 path with `AssertSummationElementMultiple % (DepthU*2) == 0`, `closeLoop` emits four close-loop instructions: `SCmpEQU32 + SCSelectB32 + SSubU32 + SCmpEQI32`. The CMS side tags all four as "LCC" via `build_idmap`'s `loopCounterCode=closeLoopMod` parameter. The SHADOW side, via `_appendCloseLoopLCCToBuilder`, only captures the `SSubU32 + SCmpEQI32` pair. Resulting SHADOW LCC=2, CMS LCC=4. The parity test at `test_capture_pipeline_checks.py:856` does not exclude LCC, so a fixture exercising this combination would fail count-parity.
**Repro:** Construct a CMS kernel with `PrefetchGlobalRead=2`, `AssertSummationElementMultiple=64` (or any other multiple of `DepthU*2`), and run `test_shadow_main_capture_categories_match_cms_subject`.
**Fix sketch:** Replace the isinstance filter in `_appendCloseLoopLCCToBuilder` with a category-driven harvest: build a temporary idMap with `loopCounterCode=closeLoopMod`, invert it, and append every leaf the inversion tags "LCC" (skipping TextBlock). This makes the LCC harvest follow the canonical schema rather than a hard-coded pair.

### Bug 2: LoopIters==1 SHADOW does not apply split_for_plr; pack categories diverge
**Severity:** MEDIUM (latent — BPG#11 has LoopIters=2)
**File:line:** `KernelWriter.py:5290` (`num_loop_iter=len(LRCodeAAllIters)`)
**Description:** `customMainLoopSchedule:dispatch.py:103-111` calls `scap.split_for_plr(LRCodeA[0])` when `numLoopIter == 1`, splitting the single-iter buckets into 2 for CMS's per-iter category schema. The SHADOW leftover walk at `KernelWriter.py:5289-5307` invokes `build_idmap(num_loop_iter=len(LRCodeAAllIters))` with the unsplit `LoopIters=1` count, tagging pack leaves as `PackA0/PackB0` only. CMS-side will tag them as `PackA0/PackA1/PackB0/PackB1` after split. Net: SHADOW vs CMS pack-category SETS differ in 1-iter kernels (CMS has 4 categories, SHADOW has 2), and counts split unevenly. The parity test would fail.
**Repro:** Construct a CMS kernel with `LoopIters=1` (e.g. small DepthU + small `MatrixInstK`) and exercise the parity test.
**Fix sketch:** Mirror `dispatch.py`'s `numLoopIter==1` handling in the SHADOW leftover walk. When `len(LRCodeAAllIters) == 1`, call `split_for_plr` on each per-iter source list, then call `build_idmap` with `num_loop_iter=2`.

### Bug 3 (NICE-TO-HAVE): Private attribute access on LoopBodyCaptureBuilder
**Severity:** LOW (code hygiene)
**File:line:** `KernelWriter.py:5326` (`for _ti in builder._instructions:`)
**Description:** The pre-skip walk accesses the private `_instructions` list of `LoopBodyCaptureBuilder` (defined at `ScheduleCapture.py:876`). If the builder is refactored to use different internal storage, this site silently breaks (or AttributeErrors loudly — depending on rename). The comment at `:5321-5323` acknowledges this ("no public accessor and adding one isn't load-bearing here"). Adding a `LoopBodyCaptureBuilder.captured_instructions()` accessor would resolve the fragility.
**Fix sketch:** Add `def captured_instructions(self) -> list: return list(self._instructions)` to `LoopBodyCaptureBuilder` and update the call site.

## Corner cases investigated

### 1-iter loop (LoopIters==1)
**Verdict:** LATENT BUG — see Bug 2 above. SHADOW leftover walk uses raw `len(LRCodeAAllIters)=1` and does not apply `split_for_plr` like CMS does. SHADOW pack tags will be `PackA0/PackB0` only; CMS will produce `PackA{0,1}/PackB{0,1}`. Not exercised by BPG#11.

### ForceUnrollSubIter enabled
**Verdict:** SAFE for the SHADOW capture path. The per-side leaf-id sets at `_loopBody:4831, 4845, 4853, 4866` (and equivalent in `_noLoadLoopBodyDefault`) are populated INSIDE the conditional `if not kernel["ForceUnrollSubIter"] or (doReadA and ...)` gates that already govern the production call. When the guard skips a localReadSwapOffsets call, no leaves are added to pointerLRCode AND no ids enter the side set — symmetric. The leftover walk and `_makeSubIterSchedule`'s pointer walk both iterate over `pointerLRCode.flatitems()`, so absence-by-skip is consistent.

### OptNoLoadLoop=True
**Verdict:** EXPLICITLY REJECTED with `NotImplementedError`. `KernelWriter.py:5213-5232` guards the SHADOW capture path: when `_captureDefaultSchedule` is set AND OptNLL conditions are met (CMS+GSU>1 with OptNoLoadLoop), the build raises before SHADOW finalize. This avoids the corner case rather than handling it. The error message is informative and the gate is correct.

### Sparse-MX enabled
**Verdict:** SAFE for current test surface; FAIL-LOUD will fire when Phase 3 adds a sparse-MX fixture. No active sparse-MX unit test fixtures exist (`Tensile/Tests/unit/test_*.py` `grep "Sparse"` finds 2 files, neither runs schedule capture). The metadata-leaf handling at `KernelWriter.py:4851-4863, 4901-4906, 3858-3863, 3905-3909` intentionally leaves leaves un-bucketed; per-side fail-loud=True will raise `CaptureCategoryMissingError` at the appropriate site. This is the design's intended discovery mechanism for sparse-MX category schema.

## Broader test surface

Ran:
```
PYTHONPATH=<wt> python3 -P -m pytest -P --ignore=test_MatrixInstructionConversion.py Tensile/Tests/unit/ --tb=line -q
```
Result: `19 failed, 1032 passed, 3 skipped, 2 xfailed, 2 errors in 20.99s`.

Failures classify as:
- `test_ScheduleCapture.py` (10 failures): Approach-A-build tests (`TestRealKernelCapture`, `TestPhase4DefaultCapture`, `TestPhase5DefaultTailCapture`, `TestDataflowGraphIntegration`, `TestPgrPlrCaptureMatrixEndToEnd`). These were failing at `f6891aab835` (the validator-branch tip, before nmsx) per round-2 verifier's reported baseline.
- `test_cms_from_default.py` (4 failures): all show `RuntimeError: ... has no pre-CMS snapshot` — Approach-A path failures, not nmsx-related.
- `test_dataflow_graph_emission_ordinal.py` (3 failures): Approach-A reference vs CMS comparison; same baseline-pattern UNKNOWN-class drift.
- `test_prologue_capture.py` (2 failures): Approach-A prologue capture failures.
- `test_cross_subiter_alu_carveout_real_kernel.py` (2 errors): collection errors involving Approach-A path.

**Verdict: matches baseline.** No new regressions from nmsx commits 418a05c96e3 and f59d6fb51df. All failures are in files that exercise the Approach-A code path (whose flaws nmsx Phase 1 explicitly does not fix and which Phase 4 will delete).

## Recommendation

**Verdict: HAS-CORRECTNESS-CONCERNS** — but the two latent bugs are precisely the class of issue the design's Phase 3 fixture-coverage requirement (`DEFAULT_SCHEDULER_REFERENCE_DESIGN.md:155`) is designed to surface. The current code is correct for the BPG#11 BPG-shape and the broader fixture surface used by the unit tests today.

**Before Step 9 (squash-merge), at minimum:**

1. **File a bead for Bug 1 (PGR=2 LCC undercount).** Reference `KernelWriter.py:2951` and `KernelWriterAssembly.py:6845-6859`. Either fix in this PR (small change: replace isinstance filter with idMap-driven harvest) or document the gap so Phase 3 fixture coverage MUST include a PGR=2 + ASEM-multiple fixture and the parity test surfaces it.

2. **File a bead for Bug 2 (LoopIters==1 split_for_plr divergence).** Reference `KernelWriter.py:5290` and `dispatch.py:103-111`. Phase 3 fixture coverage MUST include a LoopIters=1 CMS fixture and surface the gap.

3. **Confirm or add the MFMA-shape follow-up bead** mentioned in round-2 verdict as still missing (`br search mfmaIter` was empty per round-2). Without it the MFMA exclusion in the parity test is a documented punt.

**If both latent-bug beads are filed:** safe to merge. The code correctly implements the three fixes for the documented test surface; the gaps are scope-of-Phase-3 fixture-coverage rather than defects in the Phase 1 fixes themselves.

**If the latent bugs must be fixed in this PR before merge:** Bug 1 is a ~10-line edit (idMap-driven harvest in `_appendCloseLoopLCCToBuilder`); Bug 2 is a ~10-line edit (mirror `split_for_plr` in the leftover walk when `LoopIters==1`). Both are localized and would not expand the PR's blast radius.
