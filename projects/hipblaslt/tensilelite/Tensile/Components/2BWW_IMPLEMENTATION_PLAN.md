# rocm-libraries-2bww — Implementation Plan (Option C)

**Bead:** `rocm-libraries-2bww` (P0)
**Author:** opus planning subagent
**Date:** 2026-05-22
**Branch context:** `users/alvasile/fp8_fast_reference_gemm`; planning done in `/home/alvasile/rocm-libraries/.worktrees/validator_long_term_plans/projects/hipblaslt/tensilelite/`
**Status:** planning only — no production code changes.

---

## 1. Summary

The CMS schedule bodies in `Tensile/Components/CustomSchedule/gfx950/_*.py` currently mutate kernel-level flags (`MfmaInitCVgprs`, `UsePLRPack`, `UseMFMAF32XEmulation`, `UseDot2F32XEmulation`, `SwapGlobalReadOrder`) during dispatch. Per `LDM5_INVESTIGATION.md §7`, this is the root mechanism behind p39d / ldm5 / nyb5 residual classes: a non-CMS reference build that takes the `hasCustomSchedule` early-return at `dispatch.py:401-402` never sees those mutations, so reference and CMS Solutions diverge.

This plan implements **Option C**: schedule bodies stop mutating; instead they **declare expected flag values on the decorator** and **validate** the kernel's state at call time, returning `(False, None)` via the existing layout-rejection pattern when the kernel does not match. The YAML benchmark configs become the source of truth for the *YAML-tunable* flags (`UsePLRPack`, `SwapGlobalReadOrder`). The *framework-derived* flags (`MfmaInitCVgprs`, `UseMFMAF32XEmulation`, `UseDot2F32XEmulation` — none of these appear in `Tensile/Common/ValidParameters.py`) are handled CMS-side at the Solution layer via a derivation block keyed on the matched schedule's declared expectations. The 9lcs reconciliation block in `Solution.py:1984-2029` collapses into a clean "did `hasCustomSchedule` return True?" branch; the stash / pre-zero / YAML-intent-reject become dead code; the two negative-reject tests in `test_cms_flag_reconciliation.py` are deleted. The round-trip equivalence test survives and its `_SCHEDULE_CHOICE_KEYS` allowlist shrinks. The runner's `_PER_TILE_REF_FLAG_OVERRIDES` machinery is deleted at the end (Phase 5) — the runner gets the expected flag values straight from the decorator-declared `required_flags` map.

---

## 2. Design decisions (Q1–Q9)

### Q1. Runner reference-side flag source — **decision: hybrid of (a) + (b)**

The runner's `_cms_kernel_info_to_config(info)` builds a kernel config from `CMSKernelInfo` decorator metadata. Under Option C the runner must mirror the flag values the YAML would have supplied. We do **not** want option (c) (load real YAMLs) — that introduces a heavy dependency for a runner that is fundamentally a synthetic-config sweep.

**Choice:**
- Extend `CMSKernelInfo` with a `required_flags: Mapping[LayoutKey, Mapping[str, Any]]` field (where `LayoutKey = (LDSTrInst: bool, TransposeLDS: int)` keyed off the per-branch shape).
- The `RegisterSchedule` decorator owns this map (see Q2). `CMSKernelInfo` is populated from it.
- `_cms_kernel_info_to_config(info)` looks up the entry for `(info.LDSTrInst, info.TransposeLDS)` and merges its values into the generated config.
- YAML production configs separately encode the same values for the YAML-tunable subset (`UsePLRPack`, `SwapGlobalReadOrder`). The framework-derived flags are not in YAML at all; they are derived in Solution.py from the same decorator metadata, looked up via `hasCustomSchedule`.

This collapses to "Option A's declarative metadata surface + Option B's wrapped_func-driven validation" in the brief's framing.

**Why:** maximally faithful (decorator IS the truth), zero coupling to YAML for the runner, single source of truth for both Solution-side derivation and runner-side mirroring. Brittleness of hand-curated overrides goes away.

### Q2. Wiring the validation — **decision: declarative on the decorator, auto-checked by `wrapped_func` (option (b))**

The schedule body should **not** carry per-branch `if kernel.get(...) != expected: return False, None` lines. Instead:

```python
@RegisterSchedule(
    tile_config=TileConfig(128, 192, 32, 2, 1, 1, False, 0, 0, isa=(9, 5, 0)),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2],
    required_flags={
        # key = (useLDSTr, TLDS) for the branch (extended to layout if needed)
        ("TN", False, 1): {
            "UsePLRPack": True,
            "UseMFMAF32XEmulation": False,
            "UseDot2F32XEmulation": False,
            "MfmaInitCVgprs": True,
        },
        # ("NN", ...): later PRs
    },
)
def _get_schedule_128x192x32_TF32(kernel, useLDSTr, TLDS):
    ...
```

`wrapped_func` derives the layout string and `(useLDSTr, TLDS)` key, looks up `required_flags`, and checks each declared (key, value) pair against `kernel[...]`. On mismatch it returns `ScheduleMatchStatus.UNSUPPORTED_VARIANT` (which already exists in the dispatch state machine) so the dispatcher correctly stops searching once it has matched on tile/MI/wave-group/dtype.

The inner schedule body becomes purely scheduling content — no `kernel[...] = ...` writes anywhere.

**Why:** keeps validation logic out of N=41 schedule files; one bug-fix in `wrapped_func` covers the corpus; matches how Q1 wants to read the metadata. Hardcoded per-branch checks (option (a)) would be 100+ duplicated lines and re-introduce per-schedule drift risk.

### Q3. Universal-flag handling for `MfmaInitCVgprs = True` — **decision: (b) — CMS-side Solution.py derivation**

`MfmaInitCVgprs = True` is set by every successful schedule (audit §2a). Three sub-options:
- **(a)** Add `MfmaInitCVgprs: [True]` to every CMS YAML entry — high YAML churn (~36 YAML groups across the two CMS test YAMLs) AND requires adding `MfmaInitCVgprs` to `Tensile/Common/ValidParameters.py` since it currently is not a YAML-tunable param.
- **(b)** Derive once at Solution layer: after `hasCustomSchedule` resolves to True, `Solution.py` sets `state["MfmaInitCVgprs"] = True` (parallel to existing line 2040 which sets it for `UseMFMAF32XEmulation`).
- **(c)** Per-schedule `required_flags` entry. Doesn't help — still need somewhere on the non-CMS reference side to make this True, otherwise round-trip equivalence breaks.

**Choice (b).** A new line near `Solution.py:2002`:
```python
state["UseCustomMainLoopSchedule"] = 1 if hasCMS else 0
if hasCMS:
    state["MfmaInitCVgprs"] = True   # universal post-condition of CMS dispatch
```

This eliminates the universal-mutation problem with zero YAML churn and zero ValidParameters changes. The `required_flags` mechanism (Q2) handles only flags that vary per-branch.

### Q4. The four contract-violation schedules (`_128x192x64_16bit`, `_192x128x64_16bit`, `_224x320x64_16bit`, `_96x256x64_16bit`) — **decision: derives cleanly from (b) + (Q2)**

Under Q3(b), `MfmaInitCVgprs` is no longer touched by any schedule body, so the "set before layout branch" anomaly (O7 in the audit) becomes a non-issue automatically — those `kernel["MfmaInitCVgprs"] = True` lines just get deleted alongside the others.

For `_224x320x64_16bit` and `_96x256x64_16bit` which also write a second flag at function entry:
- `_224x320x64_16bit` L46: `SwapGlobalReadOrder = False` — defensive write to default (see Q7); becomes a deletion, no decorator entry needed.
- `_96x256x64_16bit` L52: `SwapGlobalReadOrder = True` — branch-conditional, lives inside `if isTN(kernel) and TLDS==1`; this becomes a `required_flags` entry keyed on the TN branch only.

Net: no special handling needed for the four contract-violation files. They migrate the same way as everyone else.

### Q5. `UsePLRPack` + `UseMFMAF32XEmulation` co-occurrence — **decision: batch them per-branch in `required_flags`**

Audit §4 confirms every `UsePLRPack` write co-occurs with a `UseMFMAF32XEmulation` write in the same branch. The plan's `required_flags` per-branch dict naturally captures this: one entry per layout/branch, multiple keys per entry. Validation is a single dict-comparison loop. No design change vs Q2.

### Q6. `UseDot2F32XEmulation = False` defensively — **decision: no validation, just delete the writes**

Audit §2e + Solution.py:624 confirm the framework default is `False` and the schedule only ever writes `False`. The writes are no-ops at the state level. Therefore:
- Drop the writes from the schedule bodies (audit §2e: 9 occurrences).
- Do **not** add `UseDot2F32XEmulation: False` to any `required_flags` entry — the default is already correct.
- Defensive rationale (someone might later change the framework default) does not justify keeping the mutation. If the default ever changes, the answer is to fix it once at the Solution layer or add a validation across all 41 schedules — both of which are better signaled by a test failure than a silently-correct mutation.

### Q7. `SwapGlobalReadOrder = False` defensively in 4 schedules — **decision: same as Q6, just delete**

Per Q6, the framework default for `SwapGlobalReadOrder` is `0`/`False` (per `Tensile/Common/ValidParameters.py:238` valid set is `[0, 1]` and Solution-side gating at lines 2077/2080 zeros it under specific conditions but the *initial* value comes from YAML, defaulting to 0 when the YAML omits it).

The four `SwapGlobalReadOrder = False` writes (`_192x320x64_16bit`, `_208x256x64_16bit`, `_224x320x64_16bit`, `_240x256x64_16bit`) are defensive resets that would only ever change a value the YAML explicitly set to `True`. Under Option C, the YAML is the source of truth: if a YAML user requests `SwapGlobalReadOrder=1` and the matched schedule does not want it, **the validation in `wrapped_func` should reject** (return `UNSUPPORTED_VARIANT`). So the correct migration is:
- Delete the four defensive writes.
- Add `"SwapGlobalReadOrder": False` to the matching `required_flags` entries — the validation will reject any YAML that sets it to True on these schedules.
- Verify that the existing CMS YAMLs do not set `SwapGlobalReadOrder` for these tile shapes (a quick grep at the start of Phase 3 confirms this; if any do, fix the YAML).

### Q8. The two skip-list schedules — **decision: migrate `_256x256x128_8bit` defensively; defer `_256x96x64_16bit_DPLB`**

- `_256x256x128_8bit` — has a YAML that exercises a related shape but with `PLR` mismatch (audit map §A). Its only mutation is `MfmaInitCVgprs = True` (audit §2a row 30) which becomes Q3(b) automatically — nothing schedule-side to migrate. **No special action.**
- `_256x96x64_16bit_DPLB` — orphan, no YAML reaches it (audit map §4). Its body is documented as mutation-free (audit §1 footnote). **No migration needed.** Optional follow-up bead: "Add YAML coverage for DPLB variant."

Therefore the audit's "skip list" concern is moot: neither schedule is affected by this work since neither writes branch-conditional flags. After migration, the audit's findings remain accurate for the unmigrated schedules: contract violations on these two persist only as the universal `MfmaInitCVgprs` write, which is collapsed at the Solution layer (Q3(b)).

### Q9. Contract-violation × skip-list intersection — **decision: not an issue**

Among the four contract-violation schedules (Q4), none is in the skip list. `_128x192x64_16bit`, `_192x128x64_16bit`, `_224x320x64_16bit`, `_96x256x64_16bit` all have YAML coverage per `CMS_SCHEDULE_TO_YAML_MAP.md §2` and all are migrated in Phase 3. No tile-level overrides for `MfmaInitCVgprs` persist after the migration.

---

## 3. Phasing

Each phase is **one PR with multiple commits**. Tests must be green at the end of each phase. Per-phase commit estimates assume the work landed in one branch with logical commit boundaries; the final PR may squash.

### Phase 0 — Baseline (no code changes)

**Goal:** confirm the baseline matches the contract from `LDM5_INVESTIGATION.md §7`.

**Tests:**
1. Unit suite must be `1033 passed / 3 skipped / 2 xfailed`:
   ```
   python -c "import os; os.chdir('/home/alvasile/rocm-libraries/.worktrees/validator_long_term_plans/projects/hipblaslt/tensilelite'); import pytest; pytest.main(['Tensile/Tests/unit/', '--ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py', '--timeout=120', '-q'])"
   ```
2. 3ija runner must show 7 CG residuals / 0 WC / 0 `CaptureConsistencyError`:
   ```
   python -c "import os; os.chdir('...'); import pytest; pytest.main(['Tensile/Tests/unit/_3ija_residual_triage_runner.py', '-s', '--ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py'])"
   ```

**Stop condition:** any mismatch — investigate before proceeding. Do not start Phase 1.

**Files touched:** none. **Commits:** 0.

### Phase 1 — Declarative `required_flags` surface

**Goal:** add the metadata plumbing without changing any schedule behavior. End-state: tests still green, schedule bodies still mutate.

**Files touched:**
- `Tensile/Components/CustomSchedule/dispatch.py` — extend `RegisterSchedule.__init__` to accept optional `required_flags` kwarg. Extend `wrapped_func` to check the per-branch entry. Extend the `_SCHEDULE_METADATA` build to copy `required_flags` into `CMSKernelInfo`.
- `Tensile/Components/CustomSchedule/shared.py` — extend `CMSKernelInfo` with `required_flags: Mapping[Tuple[str, bool, int], Mapping[str, Any]] = field(default_factory=dict)`. Key is `(layout_str, useLDSTr, TLDS)`. (Use `dict` literal at the decorator; freeze at registration time if needed.)
- `Tensile/SolutionStructs/Solution.py:1984-2029` — no change yet; we just want the surface ready.

**Commits (4):**
1. Extend `CMSKernelInfo` dataclass.
2. Extend `RegisterSchedule` to accept `required_flags`.
3. Wire `wrapped_func` to validate against `required_flags` (no-op until a decorator actually declares one).
4. Update `query_cms_kernels` / `get_cms_kernel_info_objects` to surface `required_flags` (the runner will need this in Phase 2).

**Tests:** unit suite + 3ija runner. Both must be unchanged from Phase 0.

**Rollback:** revert the phase branch; no schedule body touched, so blast radius is zero.

### Phase 2 — Pilot migration: `_128x192x32_TF32` (the ldm5 evidence schedule)

**Goal:** prove Option C end-to-end on the smallest known-good schedule. ldm5 fixture must show `cg=0` **with the `_PER_TILE_REF_FLAG_OVERRIDES` removed for this entry**.

**Files touched:**
- `Tensile/Components/CustomSchedule/gfx950/_128x192x32_TF32.py` — add `required_flags` to the decorator (TN/F/1 branch); delete the four `kernel[...] = ...` writes (L51-53, L97).
- `Tensile/Tests/common/gemm/gfx950/custom_mainloop_scheduling_tf32.yaml` — for the YAML groups that target this schedule's TN branch, add `UsePLRPack: [1]` to the ForkParams. (The TF32-XEmulation-False and Dot2-XEmulation-False are framework-derived — handled by Q3(b) / Q6, no YAML change for them.)
- `Tensile/Tests/unit/_3ija_residual_triage_runner.py` — remove the `_get_schedule_128x192x32_TF32` entry from `_PER_TILE_REF_FLAG_OVERRIDES` and extend `_cms_kernel_info_to_config` to merge `required_flags` from `info` into the generated config (single new helper call).
- `Tensile/SolutionStructs/Solution.py` — apply Q3(b) `MfmaInitCVgprs = True` if hasCMS line (this is universal and needs to land in Phase 2 so the pilot's reference-side `MfmaInitCVgprs` matches).

**Commits (5):**
1. Add `required_flags` to `_128x192x32_TF32` decorator; delete mutations from body.
2. Solution.py: derive `MfmaInitCVgprs = True` post-`hasCustomSchedule` resolution.
3. Update `custom_mainloop_scheduling_tf32.yaml` for the affected group.
4. Runner: merge `required_flags` into `_cms_kernel_info_to_config`; drop the now-unneeded `_PER_TILE_REF_FLAG_OVERRIDES` entry for `_128x192x32_TF32`.
5. Test: add a focused unit test that asserts `hasCustomSchedule(kernel_with_wrong_UsePLRPack)` returns `False` for this tile (i.e., validation works).

**Tests:** Unit suite must remain `1033/3/2` (the deleted negative-reject 9lcs tests are still alive at this point because we have not yet relaxed the 9lcs block — see Phase 4; the new validation test brings the count to `1034/3/2`). 3ija runner must show one fewer `CaptureConsistencyError`: the `_128x192x32_TF32 TN LDSTr=False TLDS=1` fixture must report `cg=0 wc=0`.

**Stop condition:** if the pilot fixture does not flip to `cg=0`, halt and root-cause before touching any other schedule.

**Rollback:** revert this phase; Phase 1 is independent.

### Phase 3 — Corpus rollout

**Goal:** migrate the remaining 36 mutation-bearing schedules (37 total minus the pilot = 36).

**Approach:** the migration follows a fixed recipe per schedule:
1. Identify the per-branch mutation lines from the audit.
2. Translate to `required_flags` entries keyed on `(layout, useLDSTr, TLDS)`.
3. Delete the mutation lines from the body.
4. For YAML-tunable flags (`UsePLRPack`, `SwapGlobalReadOrder`): edit the corresponding YAML group(s) per the schedule→YAML map. For framework-derived flags (`UseMFMAF32XEmulation`, `UseDot2F32XEmulation`): no YAML change needed — they live entirely in `required_flags` and are checked against the Solution-derived state.
5. Drop the schedule's entry from `_PER_TILE_REF_FLAG_OVERRIDES` in the runner if present.

**Recommended sub-order (group by flag class to minimize cognitive load):**

- **Phase 3a — TF32 MFMAF32XEmulation/Dot2/PLRPack cluster** (~10 schedules touching the same three flags): `_128x128x32_TF32`, `_128x128x32_TF32_plr1`, `_128x128x64_TF32`, `_128x160x64_TF32`, `_128x256x32_TF32`, `_160x128x64_TF32`, `_192x128x32_TF32`, `_192x256x32_TF32`, `_256x128x32_TF32`, `_256x256x32_TF32`, `_64x128x64_TF32`. Critically includes the three remaining ldm5 fixtures (`_192x128x32_TF32`, `_256x128x32_TF32`, `_256x192x32_TF32`).
- **Phase 3b — 16-bit SwapGlobalReadOrder=True cluster** (~10 schedules): `_160x256x64_16bit`, `_192x256x64_16bit`, `_224x128x64_16bit`, `_256x160x64_16bit`, `_256x192x64_16bit`, `_256x208x64_16bit`, `_256x224x64_16bit`, `_256x256x64_16bit`, `_320x192x64_16bit`, `_96x256x64_16bit`.
- **Phase 3c — Defensive-False writes** (4 schedules): `_192x320x64_16bit`, `_208x256x64_16bit`, `_224x320x64_16bit`, `_240x256x64_16bit`. Per Q7: just delete the writes; optionally add `SwapGlobalReadOrder: False` to `required_flags`.
- **Phase 3d — `MfmaInitCVgprs`-only schedules** (~12 schedules with no other mutation): these reduce to "delete the one line" once Phase 2's Solution-derivation lands. Includes `_128x192x64_16bit`, `_128x224x64_16bit`, `_128x256x64_16bit`, `_192x128x64_16bit`, `_224x256x64_16bit`, `_240x256x64_16bit` (overlaps 3c), `_352x192x64_16bit`, `_256x256x128_8bit`, etc.
- **Phase 3e — `_256x192x32_TF32` (multi-branch — conflicting `UseMFMAF32XEmulation`)** (audit O8): branch 1 wants `False`, branch 2 wants `True`. The per-branch `required_flags` dict naturally encodes both. This schedule is also an ldm5 fixture so it gets specific runner-side attention.

**Files touched:** every file under `Tensile/Components/CustomSchedule/gfx950/_*.py` except the pilot and the two skip-list files; the two CMS YAML files; the runner.

**Commits:** roughly one per schedule = ~36 commits. Each commit leaves all tests green. Sub-phases may bundle commits if they share a YAML edit (e.g. one YAML edit covering multiple TF32 schedules in the same group).

**Tests:** after every sub-phase, run unit suite + 3ija runner. Sub-phases are roll-back-safe individually.

**Stop conditions per sub-phase:**
- A YAML edit triggers `valid parameter check failed` — investigate: probably the schedule was matched by a YAML group that does not actually want CMS, requiring a YAML refactor or a schedule-side `required_flags` tightening.
- A 3ija runner fixture flips from `cg=0` to non-zero — root-cause; probably a missed flag in `required_flags`.
- A new fixture surfaces `CaptureConsistencyError` — likely an undocumented flag the schedule was implicitly relying on; add to `required_flags` and update audit.

**Rollback:** per-sub-phase revert. No phase blocks another except 3e depending on 3a's `_256x192x32_TF32` work (handle in 3e).

### Phase 4 — 9lcs simplification

**Goal:** simplify `Solution.py:1984-2029` now that `hasCustomSchedule` is pure (no kernel mutation visible after it returns False).

**Files touched:**
- `Tensile/SolutionStructs/Solution.py:1984-2029` — collapse the block to:
  ```python
  if state["UseCustomMainLoopSchedule"] in [-1, 1]:
      user_requested_cms = (state["UseCustomMainLoopSchedule"] == 1)
      hasCMS, _ = hasCustomSchedule(state)
      if user_requested_cms and not hasCMS:
          reject(state, printRejectionReason, "UseCustomMainLoopSchedule=1 but CMS is not supported")
      state["UseCustomMainLoopSchedule"] = 1 if hasCMS else 0
      if hasCMS:
          state["MfmaInitCVgprs"] = True   # universal CMS post-condition (Q3b)
          if state["TailloopInNll"]:
              reject(state, printRejectionReason,
                     "UseCustomMainLoopSchedule=1 is incompatible with TailloopInNll=True")
              return
  ```
  Deleted: `yaml_SwapGlobalReadOrder`/`yaml_UsePLRPack` stash, the YAML-intent reject for both flags, the post-probe pre-zero. **`hasCustomSchedule` is now safe to call without preserving these flags because schedule bodies no longer mutate.**
- `Tensile/Tests/unit/test_cms_flag_reconciliation.py`:
  - **Delete** `test_cms_rejects_yaml_swap_global_read_order`. The new behavior is: if a YAML sets `SwapGlobalReadOrder=1` on a schedule whose `required_flags` says False, `wrapped_func` returns `UNSUPPORTED_VARIANT` and the kernel falls through to non-CMS. The user-facing failure mode is "CMS did not match" rather than "loud reject" — that is the principled position now that schedule selection is purely declarative.
  - **Delete** `test_cms_rejects_yaml_use_plr_pack`. Same rationale.
  - **Keep** `test_roundtrip_no_extra_flags_matches`. Shrink `_SCHEDULE_CHOICE_KEYS` from `{UseCustomMainLoopSchedule, SwapGlobalReadOrder, UsePLRPack, MfmaInitCVgprs}` to `{UseCustomMainLoopSchedule}` only — under Option C those other three are now identical across the CMS and non-CMS paths (because the YAML carries them on both sides + Q3(b) for `MfmaInitCVgprs`).
  - **Keep** `test_cms_auto_with_yaml_flags_and_no_schedule_falls_through`. The contract still holds: auto mode with no schedule falls through. (May need a minor tweak: the test currently asserts that the small tile resolves to non-CMS; that remains true.)

**Commits (3):**
1. Solution.py: collapse 9lcs block; remove stash/pre-zero/intent-reject.
2. Tests: delete the two negative-reject tests; shrink `_SCHEDULE_CHOICE_KEYS`.
3. Documentation: add a comment block in `dispatch.py:hasCustomSchedule` noting that the function is now pure (no state mutation visible after return) per 2bww.

**Tests:** Unit suite expected count: `1031 passed / 3 skipped / 2 xfailed` (we deleted two tests; the Phase 2 new test brings net to `1032/3/2`). 3ija runner unchanged.

**Stop condition:** if the round-trip test fails for any non-allowlist key, surface the offending key — it indicates a missed flag in the corpus migration.

**Rollback:** revert all of Phase 4; the corpus migration in Phase 3 still works without 9lcs simplification (the block becomes dead-but-harmless).

### Phase 5 — Runner cleanup (the proof of completion)

**Goal:** delete `_PER_TILE_REF_FLAG_OVERRIDES` and `_build_non_cms_reference_with_state_overrides` from `_3ija_residual_triage_runner.py`. The runner's reference-side build calls the stock `build_non_cms_reference` directly, AND every ldm5 fixture (`_128x192x32_TF32`, `_192x128x32_TF32`, `_256x192x32_TF32`, `_256x128x32_TF32` × their layouts) still shows `cg=0 wc=0`.

**Files touched:**
- `Tensile/Tests/unit/_3ija_residual_triage_runner.py` — delete `_PER_TILE_REF_FLAG_OVERRIDES`, `_lookup_per_tile_overrides`, `_build_non_cms_reference_with_state_overrides`. `_exercise_one` calls `build_non_cms_reference(config, asm, isaInfoMap)` directly. The `_cms_kernel_info_to_config` from Phase 2 already merges `required_flags` into the config so the reference Solution gets identical state.

**Commits (1):** single deletion commit.

**Tests:** 3ija runner ldm5 fixtures must report:
```
_get_schedule_128x192x32_TF32 TN LDSTr=False TLDS=1   cg=0 wc=0
_get_schedule_256x192x32_TF32 TN LDSTr=False TLDS=1   cg=0 wc=0
```
…with **no override map in the runner**. This is the definitive proof of completion. Aggregate `CaptureConsistencyError` count remains 0.

**Stop condition:** if any ldm5 fixture regresses to `CaptureConsistencyError` after the override map deletion, Phase 3 missed a flag for the specific schedule. Add the missing flag to its `required_flags` entry; re-run Phase 5.

**Rollback:** revert just the deletion commit; the rest of the architecture remains intact and the override map can come back as a workaround.

### Phase 6 — Follow-up beads (optional, file-only)

**Goal:** capture follow-on work that should not block 2bww.

**Files touched:** none (just bead creation via `br`).

**Beads to file:**
1. Investigate `_256x96x64_16bit_DPLB` orphan: either add YAML coverage that exercises `DtlPlusLdsBuf=True` for 256×96×64, or remove the orphan schedule.
2. Resolve `_256x256x128_8bit` PLR mismatch with `custom_mainloop_scheduling.yaml` F8BS TN section per audit map §A.
3. Audit whether non-gfx950 `RegisterSchedule` usages (gfx1250 / future) also need `required_flags` migration. Currently the audit is gfx950-only.

---

## 4. Risks and stop conditions

### Risk R1 — Framework-derived flags are not YAML-acceptable

`MfmaInitCVgprs`, `UseMFMAF32XEmulation`, `UseDot2F32XEmulation` do not appear in `Tensile/Common/ValidParameters.py`. The plan handles this via:
- Q3(b): `MfmaInitCVgprs` derived at Solution layer, not YAML.
- Q5+Q6: the XEmulation flags only appear in `required_flags`, never in YAML; the framework derives them at `Solution.py:624-639` based on `UseF32XEmulation` + `HasMFMA`. `required_flags` validates that the derivation produced the right value for the matched schedule's branch.

**Verification step (Phase 0/1 boundary):** confirm by inspection of `Solution.py:624-639` that `UseMFMAF32XEmulation` is forced to True for `HasMFMA` (gfx950) ISAs whenever `UseF32XEmulation` is set. For the four TF32 schedules that want `UseMFMAF32XEmulation = False` (audit §2d), this means the framework default WILL NOT match — and `wrapped_func` will reject. **This is the same divergence that ldm5 documented.** The non-CMS reference picked the framework default and the CMS path overrode it.

**The plan must resolve this — and does so via the runner pathway.** The runner (under Phase 2 changes) merges `required_flags` into the generated config so the reference-side `_make_solution` builds with the same intent. But for **production** kernels (i.e., real YAML configs going through the normal pipeline), where does the override happen?

**Two viable paths:**
- (R1a) Add a Solution-side post-`hasCustomSchedule` derivation that, when CMS resolves to True, looks up the matched schedule's `required_flags` for the matched branch and applies them to `state[...]`. This mirrors Q3(b)'s pattern.
- (R1b) Make `UseMFMAF32XEmulation` and `UseDot2F32XEmulation` YAML-tunable by adding them to `ValidParameters.py` and require every CMS YAML group to declare them.

**Recommendation: R1a.** R1b would touch ~70+ YAML groups across two files and would let users mis-tune a flag the framework already derives correctly for the non-CMS path. R1a localizes the asymmetry to one Solution.py block keyed on the dispatch result.

R1a means the **`required_flags` entries serve double duty**: (1) validation when called from `wrapped_func`, (2) state-derivation when applied from Solution.py after a successful match. Both readers use the same source of truth — the decorator metadata. This is the cleanest realization of Option C.

**Where this lives in Solution.py:** extend the new Phase 4 block:
```python
if hasCMS:
    state["MfmaInitCVgprs"] = True
    # Apply matched-schedule required_flags (Q3 / R1a)
    schedule_info = hasCMS_schedule_info  # returned from hasCustomSchedule
    for flag, value in schedule_info.required_flags_for_kernel(state).items():
        state[flag] = value
```
…with the caveat that `hasCustomSchedule` currently returns only `(bool, ScheduleInfo)` — `ScheduleInfo` is the per-branch schedule content. We may need to also surface the `required_flags` entry. Two options: (a) attach `required_flags` to the returned `ScheduleInfo`, (b) extend the return tuple to `(bool, ScheduleInfo, required_flags_dict)`. **Choice (a)** — adds a `required_flags: dict = field(default_factory=dict)` field to `ScheduleInfo` in `shared.py:201`. `wrapped_func` populates it when invoking the inner function. This keeps the API stable.

**This R1 resolution path needs to be confirmed working in Phase 2's pilot before scaling.** If it does not work cleanly, fall back to R1b for the TF32 XEmulation flags only — and re-scope the bead.

### Risk R2 — A schedule's flag expectations cannot be expressed cleanly

Audit O8: `_256x192x32_TF32` has conflicting `UseMFMAF32XEmulation` between branches. The per-branch `required_flags` dict handles this naturally (one entry per layout key). **Mitigation:** Phase 3e specifically validates this multi-branch case. Stop condition: if `required_flags` key collision (same layout, conflicting flag), surface as a design defect — likely the schedule needs splitting into two `@RegisterSchedule` decorators.

### Risk R3 — A YAML group exercises multiple schedules with conflicting `required_flags`

A single YAML group might (in principle) be matched by multiple schedules with different flag expectations — but each schedule's `wrapped_func` filters on tile/MI/wave-group first. As long as no two registered schedules share all of (tile, dtype, MI, wave-group, vector_widths), there is no ambiguity. **Verification:** add a Phase 1 self-test that asserts the registry has no duplicate `(tile_config, dtype_predicate, matrix_inst, mfma_wave_group, vector_widths)` keys. If it fires, surface as design defect.

### Risk R4 — Baseline mismatch at Phase 0

Per the brief: stop and report. Do not proceed to Phase 1.

### Risk R5 — Round-trip test fails after Phase 4

The `_SCHEDULE_CHOICE_KEYS` allowlist shrinks to `{UseCustomMainLoopSchedule}`. If any other key still diverges, the missing flag indicates a Phase 3 oversight. **Mitigation:** the failure message names the key — directly identifies the missing `required_flags` entry. Easy to fix; should not block the phase.

### Risk R6 — `wrapped_func` validation slows dispatch

Dispatch iterates the registry once per `Solution.__init__`. Adding ~4 dict lookups per matched schedule is sub-microsecond. **No expected performance regression.**

### Risk R7 — Layout-probing in `RegisterSchedule._detect_supported_layouts` already calls each schedule body with mutation side effects

This probing currently invokes the inner function on synthetic kernel dicts with no impact on real Solutions (the probe dicts are discarded). After 2bww, schedule bodies are pure, so the probe is even cleaner. **No issue, mild improvement.** Worth a note in Phase 1's commit message.

---

## 5. Follow-up work

- Bead: `_256x96x64_16bit_DPLB` orphan resolution (Q8).
- Bead: `_256x256x128_8bit` PLR mismatch with F8BS TN YAML section (Q8, audit map §A).
- Bead: extend `required_flags` migration to any non-gfx950 `RegisterSchedule` usages (audit was gfx950-only — verify gfx1250 schedules in `Tensile/Components/CustomSchedule/gfx1250/` are clean).
- Bead: consider moving `MfmaInitCVgprs` from framework-state-only to YAML-tunable as a follow-on cleanup, so the Q3(b) derivation can be removed and the flag becomes uniformly user-controlled (very low priority — current Q3(b) is fine).
- Bead: add `Tensile/Tests/unit/test_cms_required_flags.py` — table-driven test that, for each registered schedule × each declared `required_flags` branch, asserts that a kernel built with mismatched flags returns False from `hasCustomSchedule`. Pinning the contract.
- Bead: confirm the audit's assertion that `UseDot2F32XEmulation` and `SwapGlobalReadOrder` defensive-False writes are no-ops — add an assertion in CI that ValidParameters defaults remain unchanged, so a future framework change cannot silently invalidate the no-write decision (Q6/Q7).

---

## 6. Open questions for the human

1. **R1a vs R1b for framework-derived flags.** The plan recommends R1a (one Solution.py block applying matched-schedule `required_flags` to `state[...]`). R1b (make the flags YAML-tunable) is the alternative. R1a is cleaner architecturally; R1b makes everything uniformly YAML-driven at the cost of ValidParameters bloat and ~70+ YAML group edits. Confirm R1a before Phase 2 starts.

2. **Should the deleted 9lcs negative-reject tests be repurposed or just deleted?** The plan deletes them. An alternative is to repurpose them as "kernel with mismatched flags + CMS=1 → resolves to UseCustomMainLoopSchedule=0 with a debug log" — which would pin a different but still useful contract (the "silent fallthrough" behavior). Worth a few minutes to discuss.

3. **Skip-list policy for `_256x96x64_16bit_DPLB` and `_256x256x128_8bit`.** Plan defers both via separate beads. Confirm this is acceptable vs. a more aggressive "delete the DPLB orphan and fix the 8bit PLR mismatch in this PR" approach.

4. **Phase 3 PR cardinality.** Plan lumps all of Phase 3 into one branch with 36 commits. Alternative: ship Phase 3a..3e as separate PRs. The branch-stack would be: Phase 1 → Phase 2 → Phase 3a → 3b → 3c → 3d → 3e → Phase 4 → Phase 5. Confirm preference (and whether the `split-branch` skill should be invoked when 2bww-prep gets too large).

5. **Branching strategy.** Plan does not specify whether 2bww should land on a new branch off `develop` or stack on the validator branch. The validator branch carries many infrastructure improvements that 2bww depends on (the `_3ija_residual_triage_runner.py`, the `_PER_TILE_REF_FLAG_OVERRIDES` machinery to delete, etc.). Recommend stacking on validator. Confirm.

6. **Coverage for `_352x192x64_16bit` NT and NN, `_192x320x64_16bit` HHS, `_224x320x64_16bit` non-TN (audit map §E).** These have thin YAML coverage. Phase 3 will not surface their absence as test failures (the runner only probes detected layouts), but adding coverage would strengthen the regression net. File as a follow-up bead.

---

*End of plan. Investigation only. No production code modified.*
