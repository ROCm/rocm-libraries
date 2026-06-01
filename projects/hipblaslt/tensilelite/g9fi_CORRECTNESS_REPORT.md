# g9fi Correctness & Soundness Report

## Verdict

**CLEAN** — ready for Step 9 squash-merge.

The MFMA exclusion removal is sound; the per-leaf-on-both-sides invariant is
empirically grounded and locked in by a real-kernel fixture test;
no further category asymmetries surfaced in the dispatch.py / `_captureSubIterToBuilder`
audit; the cwd-trap guard fires correctly; the design doc is honest;
unit-suite baseline holds exactly (`19 failed, 1040 passed, 4 skipped, 2 xfailed, 2 errors`).

No new beads filed.

## Per-correctness-question

### Q1 — Removing MFMA exclusion sound?

**SOUND.** Verified each link in the agent's chain independently.

- `removeComments` at `Tensile/Components/CustomSchedule/dispatch.py:73-78` does exactly what the agent claims: walks `module.flatitems()`, filters TextBlock/SCBranchSCC1/SNop, returns the surviving leaves as a flat list (`retModule.flatitems()`).
- `customMainLoopSchedule` calls `mfmaCode = removeComments(mfmaCode)` at `dispatch.py:123`. From that point on `mfmaCode` is a flat leaf list — the `mfmaIter` Module wrapper does NOT survive into the dispatch loop.
- The dispatch loop at `dispatch.py:235-240` iterates `for miIndex in range(-1, len(mfmaCode)):` and at line 239 tags `tag_by_origin_id[id(mfmaItem)] = "MFMA"` where `mfmaItem = mfmaCode[miIndex]` — i.e. a leaf, not a Module. `macro.add(mfmaItem)` at line 240 adds the single leaf as a direct child of the macro.
- `expand_cms_macro` at `Tensile/Components/ScheduleCapture.py:2443` walks `macro.items()` (immediate children) — but since every MFMA was added as a leaf at line 240 above, and every other category is added leaf-by-leaf via `emit_instructions` at `dispatch.py:305-324`, `macro.items()` sees only leaves.
- I searched for any path where `MfmaCodeAllIters` is tagged WITHOUT first going through `removeComments`. There is none: `MfmaCodeAllIters` flows into `customMainLoopSchedule` as `mfmaCode` (`KernelWriter.py:5050`), and the first thing the function does with it (after `removeComments`) is build `tag_by_origin_id`. No alternate tagging site for MFMAs exists.

Both sides are per-leaf by construction. Dropping the MFMA exclusion is correct.

### Q2 — Lock-in test rigor?

**RIGOROUS.** `test_shadow_mfma_count_matches_cms_subject_on_bpg11` at `test_capture_pipeline_checks.py:927-991`:

- Uses real `_make_solution`-based build (`_build_bpg11_writer_and_capture` at line 604) — NOT mocks. Same TF32-TN BPG#11 config the existing nmsx tests use.
- Asserts BOTH sides equal `48` AND asserts `shadow_mfma == cms_mfma` — three independent assertions.
- Regression to per-Module on CMS side would give `mfmaCode` length 2 (one Module per mfmaIter call per `u` for `LoopIters=2`) — `cms_mfma == 2` fails the `== 48` assert AND the cross-side equality.
- Regression to per-Module on SHADOW side (e.g. deleting the walk at `KernelWriter.py:5224-5225`) would leave the SHADOW-side MFMA-tag count at 0 (or whatever the isinstance fallback finds — but MFMAInstruction does match), so `shadow_mfma` would be the count of mfmaIter sub-leaves that are MFMAInstructions = 48 anyway via the fallback at `KernelWriter.py:2797-2798`. Wait — this means deleting the walk would NOT fail the count test on BPG#11 (where mfmaIter is pure MFMAInstructions). The test would still pass.

The walk's value (per Q3 below) is fail-loud coverage of non-MFMAInstruction sub-leaves (SNop/SWaitAlu) on kernels where mfmaIter emits them. The lock-in test does NOT exercise those — BPG#11 has pure MFMAInstruction mfmaIter. A regression that deletes the SHADOW walk would not be caught by this test on BPG#11; it would only surface on a kernel with `ExpertSchedulingMode > 0` (SWaitAlu) or `s_nop != 0` (SNop in mfmaIter). This is a coverage gap but NOT a correctness defect — the lock-in test was scoped to the MFMA-count parity invariant specifically, and that parity does hold.

The hardcoded `48` is load-bearing on the CMS side (regression there would change the count). The cross-side equality alone would only mask a co-regression where BOTH sides changed identically — implausible given the two paths' independence.

### Q3 — Per-leaf walk legitimacy?

**LEGITIMATE, but with a caveat.**

- `KernelWriterAssembly.py:8367` adds `SNop` to `imod` when `s_nop != 0`. `imod` is part of `mfmaMod` (the Module returned by `mfmaIter`).
- `KernelWriterAssembly.py:8622` adds `SWaitAlu` to `imod` when `ExpertSchedulingMode > 0`.
- `macIterCode.add(deepcopy(mfmaIter))` at `KernelWriter.py:4981` deepcopies those leaves into macIterCode.

If the walk were removed on SHADOW:
- SNop leaves: hit `isinstance(item, SNop) → category = "SNOP"` at `KernelWriter.py:2801-2802`. NOT a fail-loud trip, but the leaf gets tagged SNOP (excluded from MFMA parity).
- SWaitAlu leaves: SWaitAlu is NOT in `InstructionCategory._CLASS_NAME_TO_CATEGORY` (verified by grep). They would hit the fail-loud branch and raise `CaptureCategoryMissingError`. The agent's claim is correct: removing the walk would fail-loud on kernels with `ExpertSchedulingMode > 0` whose mfmaIter contains SWaitAlu.

So the walk is necessary for kernels where mfmaIter is not pure MFMAInstructions. It is NOT necessary on BPG#11 specifically (mfmaIter there is pure MFMAInstructions and the isinstance fallback would handle it). The walk's value is forward-coverage.

The walk does NOT cause double-tagging or other defects — see Q4.

### Q4 — Direct assignment semantic change?

**COSMETIC.** Verified empirically:

- `macIterCode = Module()` is fresh per outer iter at `KernelWriter.py:4615`.
- Each `u` iteration does `macIterCode.add(deepcopy(mfmaIter))` at `:4981`.
- The deepcopied leaves get fresh Python ids (deepcopy allocates new objects).
- `capture_idmap` (built at `:5141-5156`) sources from `LRCodeAAllIters`, `PackCodeAAllIters`, `self.codes.globalReadA/B`, `globalReadIncACode/BCode`, `self.codes.localWriteA/B`, `LRSwapA/B`, `LWSwapA/B`, `loopCounterCode=Module()`, `syncCode=Module()`, `snopCode=Module()`. None of these contain the macIterCode deepcopy results — the deepcopy targets are fresh allocations alive only via macIterCode.
- Python `id()` reuse requires the prior object to be garbage-collected. The original source modules (LRCodeAAllIters etc.) hold strong references — they're alive. The deepcopied leaves are also alive (held by macIterCode). No id collisions possible.

So `capture_id_to_cat[id(_leaf)]` cannot collide with a pre-existing entry for any leaf in macIterCode. `setdefault` vs direct assignment is semantically identical. The change is cosmetic and removes the red-flag pattern.

### Q5 — cwd-trap guard soundness?

**SOUND.** Read at `test_capture_pipeline_checks.py:54-85`.

- Comparison uses `os.path.abspath` on tree-roots computed from `__file__` (test) and `_kw.__file__` (production code). Tree-path comparison, not filename comparison.
- Edge case: if `Tensile.KernelWriter.__file__` is None (namespace package), `os.path.dirname(None)` raises `TypeError`. Tensile is a regular package with `__init__.py`, so `__file__` is always a string in practice; the edge case is theoretical.
- Error message includes both observed paths AND the literal `cd <tree>` fix command. Directly actionable.
- Verified the guard fires correctly: I ran pytest with cwd outside the worktree and PYTHONPATH set to the worktree. The guard raised with the expected diagnostic. Running from inside the worktree (with `cd` per the verifier task instructions), the guard passed silently.

### Q6 — Design doc honesty?

**HONEST.** `DEFAULT_SCHEDULER_REFERENCE_DESIGN.md:56-58` (the g9fi update) accurately describes:

- The empirical refutation of the per-Module CMS interpretation.
- The exact `removeComments`/`.flatitems()`-based flattening before tag assignment, with file:line cites.
- The SHADOW-side walk's purpose (fail-loud coverage of non-MFMAInstruction sub-leaves).
- The exclusion contract (SYNC/SNOP/SSETPRIO/SBARRIER only — no MFMA).
- Cross-references to both the categories-parity test AND the explicit MFMA-count lock-in test.

The doc no longer carries the prior "equal by construction" doc-debt — it now explains WHY (per-leaf-on-both-sides) rather than just asserting equality.

### Q7 — Corner cases (other category asymmetries)?

**NO ADDITIONAL ASYMMETRIES SURFACE.**

Audited the CMS-side dispatch and SHADOW-side `_captureSubIterToBuilder` for any path that operates on Module wrappers vs leaves asymmetrically:

- **Per-iter LR/Pack (LRA{u}, LRB{u}, PackA{u}, PackB{u})**: CMS-side flow at `dispatch.py:99-102` applies `removeComments` to each `LRCodeA[u]`/`LRCodeB[u]`/`PackCodeA[u]`/`PackCodeB[u]`, producing flat leaf lists. `emit_instructions` at `dispatch.py:305-324` walks `instModule.flatitems()` and tags each leaf with `tag_by_origin_id[id(inst)] = category`. SHADOW-side `_captureSubIterToBuilder` walks `iterCode.flatitems()` and looks up `id_to_category.get(id(item))`. Both per-leaf.

- **GRA/GRB/GRIncA/GRIncB/LWA/LWB**: same pattern — `removeComments` → leaf list → `emit_instructions` walks `flatitems()`. Per-leaf.

- **LRSA/LRSB/LWSA/LWSB (Swap chains)**: `removeComments` applied at `dispatch.py:114-121`. `emit_instructions` per-leaf. Per-leaf.

- **LCC**: `loopCounterCode = removeComments(loopCounterCode)` at `dispatch.py:122`. Per-leaf.

- **MFMA**: agent's central fix — verified per-leaf in Q1.

- **SYNC/SNOP/SBARRIER**: isinstance fallback on both sides. Already in the exclusion set.

- **nllvmcntHandling SWaitCnts**: deepcopied and added as leaves directly to macro at `dispatch.py:271-289` without a `tag_by_origin_id` entry — `expand_cms_macro` falls back to isinstance → SYNC. Per-leaf, but excluded from parity.

I did not find a single category where one side tags per-Module and the other tags per-leaf. The MFMA fix was the last asymmetry of that shape.

### Q8 — Tests pass + baseline?

**MATCHES BASELINE EXACTLY.**

```
19 failed, 1040 passed, 4 skipped, 2 xfailed, 2 errors in 21.37s
```

Plus the focused `test_capture_pipeline_checks.py` run: `35 passed, 1 skipped` — the new `test_shadow_mfma_count_matches_cms_subject_on_bpg11` is among the passing tests. The single skip is `test_loopiters1_per_category_count_parity` (legitimately skipped because the LoopIters==1 fixture isn't valid in the kernel validator; the focused-unit tests in `TestLeftoverIdmapSplitForPlrLoopIters1` cover the contract).

## Bugs found (if any)

None.

## New beads filed (if any)

None.

## Recommendation

**READY FOR STEP 9 SQUASH-MERGE.**

Caveat documented for future work (not blocking): the `test_shadow_mfma_count_matches_cms_subject_on_bpg11` lock-in test does NOT cover the SHADOW-side walk's forward-coverage value (SNop/SWaitAlu in mfmaIter on kernels with `ExpertSchedulingMode > 0` or `s_nop != 0`). If the SHADOW walk is ever removed, BPG#11 will still pass via the isinstance fallback, but a SHADOW-mode kernel with SWaitAlu in mfmaIter would `CaptureCategoryMissingError`. This is a known gap, called out explicitly in the agent's commit message ("the walk MUST exist for fail-loud coverage of non-MFMAInstruction sub-leaves the mfmaIter Module may emit") and in the design-doc update. Not a defect — the lock-in test was scoped to MFMA-count parity specifically.
