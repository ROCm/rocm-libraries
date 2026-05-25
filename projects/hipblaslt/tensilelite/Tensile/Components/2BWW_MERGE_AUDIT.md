# 2BWW Merge Audit (Final Architectural Review)

**Branch:** `users/alvasile/2bww_option_c` @ `a34a9c64c73` (21 commits ahead of
validator tip `4bdbedda281` — task brief said 22; `git log --oneline` reports 21
on this worktree; immaterial)
**Worktree:** `/home/alvasile/rocm-libraries/.claude/worktrees/agent-abc3b5f13b05ae4ab`
**Reviewer:** read-only, no code modifications
**Date:** 2026-05-22

---

## 1. Verdict

**REQUEST CHANGES (non-blocking architecturally; blocking on cleanup hygiene).**

The architectural goal is met: schedule bodies are pure for the 5 target flags
(zero `kernel[...]=` writes), `wrapped_func` validates uniformly with no allowlist,
the `_yaml_supplied_cms_flags` marker is genuinely deleted, the 9lcs block is
collapsed to its essentials, and the `_PER_TILE_REF_FLAG_OVERRIDES` test-site
patch is gone. The change is mergeable in substance.

What blocks a clean-merge **at the cleanup-hygiene level** is a cluster of
exactly the patterns the user's `CLAUDE.md` forbids: commented-out dead code
blocks left in a schedule file, identical "now applied by Solution.py" tombstone
comments fanned out across 22 schedule files, and three stale comment sites
(`shared.py`, `dispatch.py` docstring, `Solution.py` two sites,
`test_cms_flag_reconciliation.py`) that reference machinery this branch
deleted. These are 5–30 minute cleanups; none implicate the runtime contract.
Land them with the squash and the branch is clean.

---

## 2. Headline counts

| Category | Count |
|---|---|
| Hack-audit findings (A.1–A.9) | **6** (1 dead-code block, 22-site stale tombstone comment, 4 stale docstring/comment sites, 1 stale phase-tag comment) |
| Stale-bead findings (B) | **2** (`rocm-libraries-2bww` parent still open; `ldm5` close-reason cites a now-deleted runner that should be amended) |
| Missing-bead recommendations (C) | **3** (real-YAML build smoke verification; commented-out NN branch revival/deletion; "tests pass" claim drift in memos) |
| Commit-hygiene issues (D) | **2** (one commit body references a "follow-up commit" without explicit Closes/Refs metadata; final review memos are committed and now misleading) |
| Memo-state findings (E) | **3** memos require deletion or relocation; **0** memo contradicts another factually (only timing/snapshot disagreements expected from incremental reviews) |

---

## 3. A: Hack audit

### A.1. `dispatch.py` `wrapped_func` — uniform validation

**PASS** on runtime contract. Lines 725–727 of
`projects/hipblaslt/tensilelite/Tensile/Components/CustomSchedule/dispatch.py`:

```python
for _flag, _expected in branch_flags.items():
    if kernel.get(_flag) != _expected:
        return ScheduleMatchStatus.UNSUPPORTED_VARIANT, None
```

No allowlist. Comment above (713–724) explicitly states "No exceptions, no
allowlists." Matches commit 9c39b0e968b body.

**HACK** on documentation, however: the `RegisterSchedule.__init__` docstring
at lines **543–559** still says "no production-time validation cross-check is
performed today" and references "the follow-up bead for a CI linter" — those
sentences were accurate before commit 9c39b0e968b and are now actively
misleading. The 5p8k linter bead is now redundant (the runtime contract IS the
linter; 5p8k could be repointed at compile-time/load-time enforcement instead
or closed).

The same docstring at line **550–551** still tells future readers to consult
`_build_non_cms_reference_with_required_flags` for the runner side — that
helper was **deleted with the runner** in commit 103369980c4. Dangling
reference.

### A.2. `Solution.py` framework block — disable-only model

**PASS** on shape. Lines 648–666 implement exactly the truth table the
commit body for 265eeb57b01 promised: unconditional False init for
`MfmaInitCVgprs` and `UseDot2F32XEmulation`; conditional disable-only for
`UseMFMAF32XEmulation`; no enable path. The `_yaml_supplied_cms_flags`
marker is genuinely gone from `Solution.__init__`.

**HACK**: comments at **Solution.py:2016** and **Solution.py:2036** still
reference `_yaml_supplied_cms_flags` ("not in `_yaml_supplied_cms_flags`",
"in `_yaml_supplied_cms_flags`"). The marker no longer exists in the code.
A future reader will grep for it, find only these comments, and assume the
marker was an oversight rather than a deliberate post-mortem reference.
Delete or rephrase as "the marker we deleted in 2bww/265eeb…" — or just
"YAML provenance is no longer tracked; not relevant here."

### A.3. `Solution.py` 9lcs block (now ~1978–2071)

**PASS, with one historical-stash cleanup.** What remains:

- Lines 1996–2007: query `hasCustomSchedule`, set
  `UseCustomMainLoopSchedule`, reject `user_requested_cms and not hasCMS`.
- Lines 2009–2022: CMS-path post-condition — `state["MfmaInitCVgprs"]=True`,
  reject CMS + `TailloopInNll`.
- Lines 2024–2061: non-CMS path keeps the pre-2bww `UsePLRPack` gating chain
  and the `MfmaInitCVgprs = True if UseMFMAF32XEmulation` derivation.

No stash variables (`yaml_SwapGlobalReadOrder`, `yaml_UsePLRPack`), no
pre-zero of `state["UsePLRPack"]=0` / `state["SwapGlobalReadOrder"]=0`, no
R1a "apply framework-derived subset to state". The block reads as intended.

Only nit: lines 2009–2018 contain a 9-line comment paragraph explaining
*why* `state["MfmaInitCVgprs"] = True` is universal, and 2032–2036 contain
a 5-line comment explaining the non-CMS branch. Both are useful at this
moment; both will rot. Consider tightening to 2–3 lines after merge.

### A.4. `_3ija_residual_triage_runner.py` deletion completeness

**PASS.** `grep` for the runner-only helpers
(`_FRAMEWORK_DERIVED_FLAGS`, `_ref_framework_derived_overrides`,
`_build_non_cms_reference_with_required_flags`, `_PER_TILE_REF_FLAG_OVERRIDES`,
`_exercise_one`) finds **no live consumers** in
`approach_a.py`, `dispatch.py`, `Solution.py`, or any test file. The
**only** match outside docs/memos is the **stale docstring reference in
`dispatch.py:550–551`** (see A.1).

### A.5. Schedule files — `kernel[...]=` writes for 5 target flags

**ONE FINDING.** `grep -nE 'kernel\[\s*"(UsePLRPack|UseMFMAF32XEmulation|UseDot2F32XEmulation|MfmaInitCVgprs|SwapGlobalReadOrder)"\s*\]\s*='`
returns three lines, **all commented out**, in
`projects/hipblaslt/tensilelite/Tensile/Components/CustomSchedule/gfx950/_128x256x32_TF32.py:219–221`:

```python
elif isNN(kernel) and TLDS==1:
    return False, None
    # kernel["UsePLRPack"] = True
    # kernel["UseMFMAF32XEmulation"] = True
    # kernel["UseDot2F32XEmulation"] = False
    # ... ~130 more lines of commented-out NN branch body ...
```

This is **the user's "no `// removed` comments" rule violated at scale.** The
branch is dead (`return False, None` at line 218; everything below is
unreachable). Either:

1. Delete the entire commented NN branch body (~130 lines) cleanly, or
2. Resurrect it into an active schedule path with `required_flags`, or
3. Move the prose intent into a follow-up bead and delete the carcass.

This is the **most visible hack on the branch.**

### A.6. `ValidParameters.py` / `GlobalParameters.py` dead entries

**PASS.** `UseMFMAF32XEmulation` is YAML-tunable with default True, exposed
in `ValidParameters` and `defaultBenchmarkCommonParameters`. `MfmaInitCVgprs`
and `UseDot2F32XEmulation` are correctly NOT in either file. `UsePLRPack`
and `SwapGlobalReadOrder` retain the same valid-set / default. No dead
declarations.

### A.7. `test_prologue_capture.py` cross-build assertion

**PASS.** Removed assertion is documented at lines 271–278 with an explicit
rationale (no `TODO: re-add when…` marker). The remaining assertion
(prologue-content `Pack*` presence) is content-meaningful, not vacuous, but
note the docstring of
`test_whole_kernel_cms_prologue_matches_non_cms_reference` (lines 311–318)
admits the trivial `None == None` reduction holds under default config —
this is a **weakening of the pin** that the docstring acknowledges
honestly. Not a hack; a documented weakening that a future kernel-config
drift could break loudly. Acceptable as documented.

### A.8. Stale `# TODO(2bww)` / phase comments

**ONE PATTERN-LEVEL FINDING.** 22 schedule files in `gfx950/` carry an
identical 1-line tombstone:

```python
# rocm-libraries-2bww (Q3b): MfmaInitCVgprs=True is now applied by
# Solution.py post-hasCustomSchedule as a universal CMS post-condition.
```

This is **redundant** — `MfmaInitCVgprs` was never declared inside these
function bodies; the comment exists only to tell future readers "we did
something elsewhere." That is exactly the "rename `_var` / `// removed`"
anti-pattern in the user's rules. The Solution.py block already explains
this; the schedule files should not.

Also, the `dispatch.py` `hasCustomSchedule` docstring at lines 409–416
still says "Legacy (un-migrated) schedules still mutate `kernel` from
inside the body; both styles coexist during the rollout." There are no
legacy mutating schedules left on this branch (A.5 confirms). Stale.

`shared.py:78–82` and `:230–235` describe `required_flags` as something
"Solution.py applies … post-`hasCustomSchedule`" — also wrong under the
strict model (Solution.py does not apply `required_flags`;
`customMainLoopSchedule` *also* does not under strict model — only
`wrapped_func` reads them, for validation). The R1a-language scrubbing in
commit a7114d9b304 missed `shared.py`.

### A.9. New helpers that look like `_yaml_supplied_cms_flags` redux

**PASS.** No new conditional of the form "this flag should not be
touched if marked." The disable-only framework block (Solution.py:648–666)
needs no provenance marker by design; that is the whole point of the
default-True flip in 265eeb57b01.

---

## 4. B: Outstanding-bead audit

### Open beads inspected (via `br show <id> --json`)

| Bead | P | Status | Title (truncated) | 2bww impact |
|---|---|---|---|---|
| `2bww` | 0 | **open** | Stop CMS schedule bodies from mutating kernel-level flags | **Should be closed at merge.** The DoD ("`_PER_TILE_REF_FLAG_OVERRIDES` deletable; ldm5-class drops to 0") is met; the bead description even names the now-deleted runner. |
| `ldm5` | 0 | closed | CaptureConsistencyError on 4x4x4 mfma TF32 fixtures | **Close-reason references `_3ija_residual_triage_runner.py` and `_PER_TILE_REF_FLAG_OVERRIDES`** as the test-site patch. Both are deleted by 2bww. Close-reason still accurate as history but should get a follow-up comment: "deletions tracked under 2bww/103369980c4 and 4b6d6520bc4." |
| `4czr` | 3 | closed | Promote MfmaInitCVgprs framework default to True for HasMFMA ISAs | Accurate; close-reason cites the implementing commit. ✓ |
| `9lcs` | 2 | closed | Reconcile UseCustomMainLoopSchedule=0 vs =1 flag handling | Closed historical; 2bww simplified the block significantly but didn't fundamentally re-open the contract. ✓ |
| `hcug` | 3 | open | `_256x96x64_16bit_DPLB` orphan resolution | Bead description cites `CMS_SCHEDULE_TO_YAML_MAP.md §4` — **memo doesn't exist on this branch.** Description otherwise still actionable. |
| `v01w` | 3 | open | `_256x256x128_8bit` PLR mismatch | Same — cites `CMS_SCHEDULE_TO_YAML_MAP.md §5A` which isn't on this branch. The actual concern (TileConfig PLR=0 vs YAML PLR=[1]) may be partially addressed by the F8BS group deletion in commit bb8112930063; **needs re-triage**. |
| `t4gl` | 3 | open | Audit non-gfx950 RegisterSchedule usages | Description correctly scopes 2bww as gfx950-only. Still actionable as-is. ✓ |
| `gz0k` | 3 | open | Thin YAML coverage | Cites `CMS_SCHEDULE_TO_YAML_MAP.md §5E` which isn't on this branch. Concern (`_352x192x64_16bit` has no NN/NT coverage; etc.) still actionable; **memo reference is stale**. |
| `5p8k` | 2 | open | CI linter for YAML vs decorator `required_flags` consistency | **Partially obsolete.** The runtime `wrapped_func` uniform-validation contract effectively IS the linter (any YAML/decorator drift produces `UNSUPPORTED_VARIANT` at build time → 9lcs reject loudly via `user_requested_cms`). The bead's value is reduced to "produce a more diagnostic error message at YAML-load time." Re-scope the bead or close. |
| `p39d` | 1 | open | GR OrderInverted residual class | **Not addressed by 2bww** (confirms expectation in task brief). 2bww's strict model doesn't touch the GR scheduling order; still a Q2-class residual. Description remains accurate. ✓ |
| `3ija` | 0 | in_progress | Triage compare_graphs/wait-coverage residuals | **The runner this bead launched was deleted in commit 103369980c4.** The dependent beads (`p39d`, `zvzu`, `6hk3`, `jmfp`, `mnzh`, `czby`, `aixt`) are still alive. **3ija itself needs an update**: either close ("triage complete; residuals tracked under dependents") or amend with a comment that the runner is gone. |

### B.1. summary

- **`rocm-libraries-2bww` open after merge** — should be closed (top
  priority among bead actions).
- **3 follow-up beads (`hcug`, `v01w`, `gz0k`) reference an audit memo
  (`CMS_SCHEDULE_TO_YAML_MAP.md`) that does not exist on this branch.**
  Without the memo, the section references are dead links.
- **`5p8k` is partially obsolete** under uniform validation.

### B.2. `3ija` parent triage status

`3ija` is `in_progress`. Its `dependents` field lists `ldm5` (closed) and 7
open follow-ups. 2bww closed `ldm5` (and `4czr`). 2bww did not close
`p39d`, `zvzu`, `6hk3`, `jmfp`, `mnzh`, `czby`, or `aixt`. **3ija's
description correctly anticipates closing as residuals split out into
sub-beads — no obsolescence.** But the `_3ija_residual_triage_runner.py`
machinery the bead's comment §1 cites is gone; a one-line comment update
would close the loop ("runner deleted in 2bww/103369980c4; triage
inventory frozen in `3IJA_RESIDUAL_TRIAGE.md`").

### B.3. Obsolete-but-not-updated descriptions

- `hcug`, `v01w`, `gz0k`: stale memo cross-reference (`CMS_SCHEDULE_TO_YAML_MAP.md`).
- `5p8k`: contract changed; bead scope shrunk.
- `2bww` itself: DoD met; close after merge.

---

## 5. C: New-bead audit + recommendations

### C.1. Recently-created beads in the 2bww timeline

The 5 beads `hcug`, `v01w`, `t4gl`, `gz0k`, `5p8k` were all filed by the
2bww work (all created 2026-05-25; all P2 or P3). All correctly cite 2bww
as parent/context **in their description body** even though `dependencies`
field is empty. Acceptable but slightly weaker than using `br dep add` to
formalize.

### C.2. Accuracy + priority

- `hcug`, `v01w`, `gz0k`: priority P3 is correct (cosmetic / dispatch
  surface narrowing). Stale memo references — see B.1.
- `t4gl`: P3 correct (no current non-gfx950 schedules exist; preventive).
- `5p8k`: P2. **Should be re-scoped** (see B.1 + C.3 below).

### C.3. Missing beads — should-file

1. **Real-YAML smoke verification.** Commit `5dee991c33c`'s body says
   "YAML build verification deferred to next commit (real Tensile
   invocation)" and **no subsequent commit confirms the smoke ran.**
   `bb811293006` references the same `custom_mainloop_scheduling.yaml`
   stream-of-failures only obliquely. Suggest filing a P2:
   "Run `Tensile --build-only` against
   `custom_mainloop_scheduling*.yaml` on gfx950 post-merge and pin the
   pass." Without this, the YAML cleanup is unit-suite-verified only.

2. **Delete the dead NN branch in `_128x256x32_TF32.py`.** Lines
   217–344 of that file are a `return False, None` followed by ~130 lines
   of commented-out code. Either (a) revive into an active branch with
   `required_flags`, or (b) delete the carcass + the now-redundant
   tombstone comment in `_128x256x32_TF32.py:219–221`. Suggest filing a
   P3 cleanup bead — or simply land this as part of the pre-merge cleanup
   (preferred; it's small).

3. **Rewrite the "Q3b: MfmaInitCVgprs is now applied by Solution.py"
   tombstone comments across 22 schedule files.** Suggest filing a P3
   cleanup bead — or fold into the pre-merge cleanup (preferred).

### C.4. Re-scope `5p8k`

Either close as obsolete (runtime contract enforces the same thing) or
re-scope to: "YAML-load-time linter producing diagnostic messages naming
the exact YAML group + offending flag, instead of the runtime
`UNSUPPORTED_VARIANT` reject which fires later in the build."

---

## 6. D: Commit hygiene

### D.1. `git log --oneline 4bdbedda281..HEAD`

- **21 commits** (task brief said 22; one commit may have been counted
  twice in the brief — log shows 21).
- **0 merge commits.** ✓
- **0 `--no-verify` markers** in commit bodies. ✓
- **0 WIP / fix-typo commits.** Every commit is plan-phase-named or
  describes a concrete delta. ✓

### D.2. Commit-message style

- All 21 commits reference `rocm-libraries-2bww`. ✓
- Subject lines are concise (≤ 80 col); bodies are thorough (sometimes
  excessive — `bb811293006`'s 90-line body would scare any reader, but the
  content is correct).
- **One hygiene nit:** `9c39b0e968b`'s body says "See follow-up commit
  '2bww: update synthetic test configs to satisfy schedule required_flags'
  for the test fixes that restore 1033/3/2." There's no explicit
  `Refs:` / commit-hash pointer. The reader has to trust the prose.
  Acceptable; not blocking.

### D.3. Scope creep

Files changed (51): `Common/*` (2), `CustomSchedule/dispatch.py` (1),
`CustomSchedule/shared.py` (1), `CustomSchedule/gfx950/*.py` (35),
`SolutionStructs/Solution.py` (1), 2 YAMLs, 9 test files, 1 deletion
(`_3ija_residual_triage_runner.py`). **No incidental noise.** All edits
sit inside the documented 2bww + 4czr scope.

### D.4. Squash recommendation

Per user's `feedback_squash_at_merge.md`, this branch should be squashed
at merge. The 21-commit history is useful for review but the final commit
message should be a single curated narrative — the phase-ordered story is
captured in the bead and the memos.

---

## 7. E: Memo state

### E.1. Memos under `Tensile/Components/`

| Memo | State | Recommendation |
|---|---|---|
| `2BWW_REVIEW.md` | Pre-strict snapshot (10 commits ahead). HEAD has moved 11 commits past. "Verdict: APPROVE WITH MINOR FIXUPS" reflects an earlier branch tip. | **Delete or move to a historical area.** A future reader will read it as current. |
| `2BWW_STRICT_REVIEW.md` | Mid-strict snapshot (13 commits ahead, HEAD `3c9af66075d`). HEAD has moved 8 commits past. Still claims "framework derivation honors YAML via `_yaml_supplied_cms_flags`" — this is now FALSE (marker was deleted in 265eeb57b01). | **Delete.** It actively misleads on the YAML-tunable contract that no longer exists. |
| `2BWW_FINAL_REVIEW.md` | Pre-strict-validation snapshot (`bb811293006` tip). **Verdict: REQUEST CHANGES (blocking) — claims two commits aren't on the branch.** Those commits ARE on the branch now. | **Delete.** This is the highest-risk memo — its verdict is the opposite of the actual final state. |
| `LDM5_INVESTIGATION.md` | Pre-2bww investigation memo. References `_PER_TILE_REF_FLAG_OVERRIDES` as the test-site patch — now deleted. Description otherwise still useful as historical record. | **Add a postscript section** noting which prescriptions are now obsolete, OR move to a historical area. |
| `YAML_USECMS_CLEANUP_PLAN.md` | Plan memo whose decisions are baked into commit `bb811293006`. Now retroactive history. | **Delete** (per user rule: "Should any memo be deleted (e.g., a planning memo whose decisions are now baked into commits)?" — yes). |
| `3IJA_RESIDUAL_TRIAGE.md` | Triage inventory snapshot. Still useful as the canonical residual catalog. The runner is gone but the catalog stands. | **Keep**; add a one-line note that the runner that produced it was retired. |

### E.2. Memo contradictions

No factual contradiction between memos; each is internally consistent
relative to its snapshot. The contradictions are between any of the
3 `2BWW_*_REVIEW.md` memos and the **current code state** — those memos
were never updated to track the branch tip past their review dates.

### E.3. Brief checklist

- `2BWW_FINAL_REVIEW.md` actively misleads — **must go** (blocking).
- `2BWW_STRICT_REVIEW.md` cites a deleted marker as current contract —
  **must go** (blocking).
- `2BWW_REVIEW.md` is the most innocent of the three but still
  pre-strict — **should go** (non-blocking).
- `YAML_USECMS_CLEANUP_PLAN.md` is a planning memo whose decisions are
  in the commit log — **should go** (non-blocking).
- `LDM5_INVESTIGATION.md` is useful history; postscript or relocate.
- `3IJA_RESIDUAL_TRIAGE.md` is useful history; one-line note.

---

## 8. Pre-merge fixups

### Blocking (must land before merge / as part of the squash):

1. **Delete the dead commented-out NN branch in
   `gfx950/_128x256x32_TF32.py` (lines 217–344).** Single largest hack on
   the branch.
2. **Delete the 22 identical "Q3b" tombstone comments** in `gfx950/_*.py`
   (one-line each). The Solution.py block already explains the design;
   the schedule files should not.
3. **Update stale docstrings/comments to match strict-2bww:**
   - `dispatch.py:543–559` (RegisterSchedule docstring — claims no
     validation, claims deleted runner-helper consumes it)
   - `dispatch.py:409–416` (`hasCustomSchedule` docstring — claims legacy
     schedules still mutate)
   - `shared.py:78–82` and `:230–235` (`required_flags` comments — claim
     Solution.py applies them)
   - `Solution.py:2016` and `:2036` (`_yaml_supplied_cms_flags` refs to a
     deleted marker)
   - `test_cms_flag_reconciliation.py:94` (same)
4. **Delete `2BWW_FINAL_REVIEW.md`, `2BWW_STRICT_REVIEW.md`,
   `2BWW_REVIEW.md`, `YAML_USECMS_CLEANUP_PLAN.md`** (or move all four to
   a historical-archive area). The first two actively mislead.

### Non-blocking (post-merge cleanup):

5. Tighten the long comment paragraphs in `Solution.py:2009–2018` and
   `:2032–2036` to 2–3 lines each.
6. Add a postscript to `LDM5_INVESTIGATION.md` and a one-line note to
   `3IJA_RESIDUAL_TRIAGE.md` noting which prescriptions are now obsolete.

---

## 9. Post-merge follow-ups

1. **Close `rocm-libraries-2bww`** with a close-reason citing the squash
   commit hash + the DoD bullets met.
2. **Run real-YAML build smoke** (`Tensile --build-only` on
   `custom_mainloop_scheduling*.yaml` for gfx950). File a P2 bead if you
   need a regression pin; close immediately if it passes.
3. **Update the 3 follow-up beads (`hcug`, `v01w`, `gz0k`)** to remove
   the dead `CMS_SCHEDULE_TO_YAML_MAP.md` cross-references and replace
   with the surviving knowledge (likely an inline summary).
4. **Re-scope or close `rocm-libraries-5p8k`** in light of uniform
   `wrapped_func` validation.
5. **Comment on `rocm-libraries-3ija`** that the runner is retired;
   either close (triage complete) or update status.
6. **Add a comment to `rocm-libraries-ldm5`** noting that the
   `_PER_TILE_REF_FLAG_OVERRIDES` test-site patch cited in its
   close-reason is now deleted by 2bww (commits 103369980c4 + 4b6d6520bc4).
7. **Re-triage `rocm-libraries-v01w`** — the F8BS PLR-mismatch concern
   may be partially addressed by the F8BS group deletion in
   `bb811293006`.

---

## 10. Closing note

The architectural target is met cleanly; the strict model is real, not
posed; the test gates pass at the same counts as the validator tip. What
this audit catches are the seams between the iterative reviews — five
review/plan memos that captured intermediate decisions, two stale
comment-only references to deleted machinery, and the 22-site tombstone
pattern across schedule files. None of these alter behavior; all of them
violate the user's standing rule "Avoid backwards-compatibility hacks
like renaming unused `_vars`, re-exporting types, adding `// removed`
comments for removed code."

The fixups in §8 are ~30 minutes of edits; do them in the squash and
this branch lands clean.
