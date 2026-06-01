# Round 2 verdict (post fix commit f59d6fb51df)

## Verdict
**CLEAN** (with one explicit caveat: a follow-up bead for the MFMA-shape divergence MUST be filed before declaring the punt-rule satisfied; see Issue 2 below).

## Per-issue status

### Issue 1 (was BLOCKING): RESOLVED

The `inst_cat.name` fallback for SHADOW is removed. At `KernelWriter.py:2820-2855` the code now raises `CaptureCategoryMissingError` unconditionally when `fail_loud_on_missing_category=True` AND the leaf is neither in `id_to_category` nor in `_registry_category_to_tag` (which now covers ONLY MFMA/SYNC/SBARRIER/SNOP/SSETPRIO per `:2776-2782`). The legacy `inst_cat.name`-fallback survives only inside the `else` arm at `:2872`, reachable solely when `fail_loud_on_missing_category=False`.

The agent's claim that the Approach-A else arm is "load-bearing until Phase 4" is verified:
- `build_non_cms_reference` (Approach-A entry) is reachable today from `test_approach_a_non_cms_reference.py:100,202`, `test_oplb_register_naming_minimal.py:181`, `test_prologue_capture.py:366`, `test_dataflow_graph_emission_ordinal.py:610`, `test_p39d_gr_orderinverted_minimal.py:174`, `test_capture_pipeline_checks.py:632` (six active test consumers).
- Approach-A flips `_captureNonCmsBuild` and enters the call site at `KernelWriter.py:4982-5061` whose explicit `capture_fail_loud_on_missing_category=False` at `:5060` reaches the else arm at `:2872`.
- Therefore the else arm is NOT currently dead. The agent's preservation is correct, not bloat. The TODO at `:2868-2871` explicitly ties the else arm's lifetime to the Approach-A call site's deletion in Phase 4.

### Issue 2 (was BLOCKING): RESOLVED-WITH-CAVEAT

**MFMA fail-loud coverage:** `KernelWriter.py:5172-5173` registers every leaf in the deepcopied `macIterCode` under the "MFMA" tag, so the fail-loud raise no longer fires on mfmaIter sub-leaves. Verified.

**MFMA-shape count divergence:** The agent honestly documents this in the comment at `KernelWriter.py:5142-5171` ("Asymmetry note: this still produces SHADOW MFMA counts ≫ CMS MFMA counts...") AND in the test exclusion docstring at `test_capture_pipeline_checks.py:819-842` (per-leaf-vs-per-Module structural shape divergence). I verified the agent's CMS-side characterization independently: `expand_cms_macro` at `ScheduleCapture.py:2443` walks `macro.items()` (immediate children), `dispatch.py:240` adds the entire mfmaIter Module as one item with origin id tagged "MFMA", and `:2490-2497` looks up the Module-id and gets "MFMA" without recursion. The structural divergence is real.

**MFMA was named in design line 57** ("Per-category counts are equal by construction on data-flow categories (LR/LW/GR/**MFMA**/CVT_PACK/MIDDLE_PACK), modulo SYNC/SNOP") and design line 113 acceptance only excludes SYNC/SNOP. The exclusion of MFMA from the parity test therefore IS a deviation from the literal design wording.

However: closing the divergence requires teaching `_captureSubIterToBuilder` to atomic-treat the mfmaIter Module (a non-trivial structural intervention that materially changes the walk model). This is genuinely larger than nmsx Phase 1's three documented capture-window/scope/walk-coverage fixes. Calling it "out of nmsx Phase 1 scope" is defensible.

**The remaining gap:** the agent said in the self-report "worth a follow-up bead" but **DID NOT FILE ONE.** `br search mfmaIter` and `br search atomic-treat` both return empty. This is the user's "don't punt work for later" pattern — declaring something out of scope without docketing it means it gets forgotten.

**Recommendation:** before declaring CLEAN final, file a P1 bead capturing the per-leaf-vs-per-Module SHADOW MFMA atomic-treatment requirement, referencing `KernelWriter.py:5142-5171` and `test_capture_pipeline_checks.py:819-842`. Without the bead the punt-flag flips on.

### Issue 3 (was BLOCKING): RESOLVED

`tensilelite/conftest.py` does not exist. The fix-commit's stat output shows `projects/hipblaslt/tensilelite/conftest.py | 59 ------` (full deletion). No replacement monkey-patching in `Tensile/Tests/unit/conftest.py` (read in full: only the legitimate `isa_infrastructure` session fixture) or `Tensile/Tests/conftest.py` (auto-discovered). `pytest.ini` is clean — only markers + xunit junit config. No `pyproject.toml` test plugin that re-introduces the patch.

### Issue 4 (was SHOULD-FIX): RESOLVED

All four cited sites no longer pre-register sparse-MX metadata leaves into `pointer_lrs_leaf_ids_B`:
- `KernelWriter.py:3852-3873` (NLL/NGL swap offsets): comment cites the design's fail-loud contract; the metadata leaves are emitted into `pointerLRCode` but NOT recorded into any per-side set.
- `KernelWriter.py:3899-3909` (NLL/NGL init pointers): same pattern.
- `KernelWriter.py:4851-4863` (mainloop swap offsets): same pattern.
- `KernelWriter.py:4897-4904` (mainloop init pointers): same pattern.

Each has an explicit comment that names the canonical `LRMetadata{u}` schema, points at the design's fail-loud contract, and describes what a real per-side metadata bucket would look like when a sparse-MX fixture lands in Phase 3. Honest scoping.

### Issue 5 (was SHOULD-FIX): RESOLVED

Defaults flipped to `True`:
- `KernelWriter.py:989` (`_makeSubIterSchedule`): `capture_fail_loud_on_missing_category=True`.
- `KernelWriter.py:2712` (`_captureSubIterToBuilder`): `fail_loud_on_missing_category=True`.

Explicit `False` audit (`grep -rn fail_loud_on_missing_category=False`):
- `KernelWriter.py:5060` — the Approach-A call site, with a Phase 4 TODO at `:5049-5059` explicitly tying the explicit-False's lifetime to the Approach-A call site's deletion. The ONLY production explicit-False.
- `test_capture_pipeline_checks.py:982` — the test that exercises the silent-fallback canary for the Approach-A path. Load-bearing.

Three production call sites:
- `:4026` (NLL/NGL SHADOW): `True`.
- `:5060` (Approach-A): `False` with Phase-4 TODO.
- `:5197` (SHADOW main_loop): `True`.

### Issue 6 (was NICE-TO-HAVE): RESOLVED

`test_shadow_main_capture_contains_per_subiter_packs` at `test_capture_pipeline_checks.py:727-786` now asserts both set membership (lines 762-771) and per-category COUNT equality (lines 773-786). The docstring explicitly calls out the count-truncation regression Fix 2 was written to prevent. The count assertion comes second so the set-coverage failure mode produces a cleaner error.

The broader parity test exclusion at `:856` is now `{"SYNC", "SNOP", "SSETPRIO", "SBARRIER", "MFMA"}` — trimmed from round 1's 10 categories to 5. Four are design-§3-legitimate scheduler-inserted, one (MFMA) is the structural-shape divergence documented at length above.

## New concerns in round 2

### Concern: MFMA-shape follow-up bead not filed

Already covered under Issue 2. The agent's self-report says "worth a follow-up bead"; no bead exists per `br search mfmaIter` / `br search atomic-treat` (both empty). This is the punt anti-pattern unless and until a bead is filed.

### Concern: design line 57 names MFMA as "equal by construction"

The design literally says counts are equal by construction on MFMA. The agent's exclusion documents the structural reason (per-leaf vs per-Module walk) thoroughly and the construction-equality claim was likely written before the per-leaf flattening was understood. The design might need a v6 minor amendment ("MFMA counts equal by construction MODULO walk-model — per-leaf SHADOW vs per-Module CMS"). This is a documentation-debt observation, not a new code defect.

### No new hacks observed

Read every diff hunk between `418a05c96e3` and `f59d6fb51df`. No new defensive try/except wrappers; no new exclusion lists; no new fallback paths; no new comment euphemisms. Comments are uniformly principled and name design citations or call out scope boundaries explicitly.

### Test suite baseline

Ran full unit suite: `19 failed, 1032 passed, 3 skipped, 2 xfailed, 2 errors in 20.88s`. This matches the agent's reported pre-existing baseline (19 + 2). The 19 failures are in `test_ScheduleCapture`, `test_cms_from_default`, `test_dataflow_graph_emission_ordinal`, `test_prologue_capture`, `test_cross_subiter_alu_carveout_real_kernel` — none touch the fix-commit's changed surface (KernelWriter.py:_captureSubIterToBuilder, _makeSubIterSchedule, _loopBody, _noLoadLoopBodyDefault). Spot-confirmed: fix-commit stat shows only KernelWriter.py + test_capture_pipeline_checks.py + conftest.py-deletion. No NEW failures introduced.

### Commit message honesty

Commit message at `f59d6fb51df` is honest: explicitly names the "graceful fallback" as a meta-issue, cites every fix to the verifier's concern numbers, calls out the MFMA structural divergence in fix 2's paragraph (not buried), describes the Approach-A explicit-False as legacy-tolerance-with-Phase-4-TODO. No marketing language; no euphemisms for the MFMA exclusion. Reproduces the baseline-failure count accurately.

## Recommendation

**Step 1 (required before final-CLEAN):** file a P1 bead capturing the per-leaf-vs-per-Module SHADOW MFMA atomic-treatment work. Reference `KernelWriter.py:5142-5171` and the design line 57 "equal by construction" wording. This converts the test-exclusion + comment from "punt" to "tracked deferred work."

**Step 2:** dispatch the correctness verifier (Step 6). All six previously-blocking issues are resolved at the code level. The MFMA-shape bead can be filed in parallel with the correctness verification.

If the user requires zero deferred work before merging Phase 1, the alternative is to restructure `_captureSubIterToBuilder` to recognize Module-level MFMA tags and skip flatitems descent for those — a single targeted patch that is, however, larger than the three fixes nmsx was scoped for and would expand Phase 1's blast radius into the walk-model. The cleaner sequencing is: file the bead, ship Phase 1, address mfmaIter atomic-treatment in a follow-up bead.

---
---

# Round 1 verdict (post implementation commit 418a05c96e3) — archived

## Verdict
**NEEDS-REVISIONS**

Three plan-level deviations cluster around a single anti-pattern: the
implementation introduces a graceful "fail-soft" fallback to enum-name tags,
then writes a parity test whose exclusion list silently masks every category
that fallback produces. Each piece in isolation looks defensible. Together
they re-create the silent-UNKNOWN slip-through that the design's fail-loud
contract was written to prevent — just with cosmetically different names.

Additional concerns: the SHADOW-vs-CMS count-parity test does not actually
assert per-category COUNT equality on the PackA{u}/PackB{u} categories that
Fix 2 was supposed to land (it tests only category-name SET coverage); the
mfmaIter deepcopy defect is documented and punted to a future bead but
its presence inside SHADOW under the bare names CVT_PACK/MIDDLE_PACK/LR is
load-bearing for the fallback-bloat above; and the root-level
`tensilelite/conftest.py` is local-dev plumbing for a wrong editable install
that should not land on the branch.

## Implementation summary
- `KernelWriter.py` + `ScheduleCapture.py` — three SHADOW capture-window/scope
  fixes per design v5 §4 Phase 1: (1) compute `closeLoopMod` once at the
  end-of-loop finalize site (line 5169-5171) and harvest LCC via existing
  `_appendCloseLoopLCCToBuilder` before `builder.finalize()` (line 5294-5295);
  (2) leftover walk over `pack[*]/packPre[*]` at SHADOW finalize site (line
  5184-5288) gated by xbi0/flpk invariants; (3) per-side
  `pointer_lrs_leaf_ids_A/B` / `pointer_lws_leaf_ids_A/B` sets recorded at
  swap-offset call sites in `_loopBody` (line 4399-4410, 4717-4842) and
  `_noLoadLoopBodyDefault` (line 3431-3442, 3734-3801), threaded into
  `_makeSubIterSchedule` via new `capture_pointer_side_map` kwarg
  (KernelWriter.py:1043-1073).
- New `CaptureCategoryMissingError` in `ScheduleCapture.py:117-139`. Raised
  in `_captureSubIterToBuilder` (KernelWriter.py:2808-2829) but ONLY when
  the new `fail_loud_on_missing_category=True` kwarg is set, AND ONLY when
  `category_of_class_name(...)` returns None.
- New unit tests file `test_capture_pipeline_checks.py` (7 new tests in
  `TestShadowCaptureNmsxFixes` + `TestShadowCaptureFailLoudOnUnknownCategory`).
- New root-level `tensilelite/conftest.py` (59 lines) that mutates `sys.path`
  and patches an editable-install finder in `/home/alvasile/venv`.

## Concerns

### Concern 1: "Graceful fallback" to enum name re-introduces silent slip-through, just with cosmetically different tags
**Severity:** HIGH

**Evidence:**
- KernelWriter.py:2925-2926: when the registry recognizes the class but its
  enum has no pre-mapped SHADOW tag, the agent falls back to
  `category = inst_cat.name` (CVT_PACK, MIDDLE_PACK, LR, LW, GR…).
- KernelWriter.py:2906-2912: `_registry_category_to_tag` covers only
  `MFMA/SWAIT/SBARRIER/SNOP/SSETPRIO`. Everything else routes to enum-name.
- Design v5 §4 Phase 1 (DEFAULT_SCHEDULER_REFERENCE_DESIGN.md line 108):
  *"For all leaves: if the SHADOW capture encounters a leaf with neither an
  idMap entry nor a registered class-name (per
  `InstructionCategory._CLASS_NAME_TO_CATEGORY`), raise
  `CaptureCategoryMissingError`."*

**Why it's a concern:** the design's fail-loud sentence is ambiguous between
"any registry entry" (agent's reading) and "a category that maps to a
canonical SHADOW tag" (stricter reading). On the strict reading the
graceful fallback is a workaround. But the load-bearing question is:
**does CMS ever produce a tag like "CVT_PACK" / "MIDDLE_PACK" / "LR" / "LW" / "GR" as a bare enum name?** No — CMS uses per-iter `LRA{u}`/`LRB{u}`/
`PackA{u}`/`PackB{u}` (ScheduleCapture.py:1021, 1025, 1098) and per-side
`LRSA`/`LRSB`/`LWSA`/`LWSB`/`LWA`/`LWB`/`GRA`/`GRB` (ScheduleCapture.py:
1068-1073). The bare names ONLY exist on SHADOW side under the agent's
fallback. So any leaf the fallback captures is *guaranteed* to mismatch CMS
on per-category counts. The fallback does not "still get the leaf captured
usefully" — it gets the leaf captured under a tag that **provably cannot
match the CMS-side schema**. That defeats the entire purpose of capturing it
for SHADOW-vs-CMS comparison.

The honest behavior is one of:
(a) raise — the strict reading of design v5; producer-site bug surfaces loudly.
(b) re-tag into the correct per-iter/per-side category via the producer's
   own per-iter knowledge (i.e. extend Fix 2's leftover walk or Fix 3's
   per-side tracking to cover these leaves rather than letting them fall
   through to enum-name).

The agent's commit message acknowledges this ("These still might mismatch
the CMS-side parallel-silent-UNKNOWN for the same leaves, but that's a
CMS-side issue to address...") — but the issue is not symmetric: CMS does
NOT produce these bare-name tags at all, so the divergence is one-sided
SHADOW-only noise.

**Recommended fix:** delete the `inst_cat.name` fallback branch (KernelWriter.py:2924-2926). Either tighten to raise, OR — preferred — extend
Fix 2's coverage so mfmaIter deepcopy sub-leaves (the actual case that
triggers this path) are tagged via the same per-iter PackA{u}/PackB{u}
machinery that the rest of the leftover walk uses. That eliminates BOTH
this fallback AND Concern 2 below as a single principled fix.

---

### Concern 2: mfmaIter deepcopy defect is punted, and the parity-test exclusion list hides the symptom
**Severity:** HIGH (this is the "punt" anti-pattern the user explicitly called out)

**Evidence:**
- KernelWriter.py:4877-4889: SHADOW deepcopy of mfmaIter is described in
  the commit body as producing new ids that "fall through to the registry
  fallback (CVT_PACK/LR)." The agent leaves this in place.
- test_capture_pipeline_checks.py:398-399 (parity test exclusion list):
  ```
  excluded = {"SYNC", "SNOP", "SSETPRIO", "SBARRIER", "MFMA",
              "CVT_PACK", "MIDDLE_PACK", "LR", "LW", "GR"}
  ```
- The first four (SYNC/SNOP/SSETPRIO/SBARRIER) are legitimate per design §3
  (scheduler-inserted). The remaining six are precisely the categories
  produced by the Concern-1 graceful fallback.
- Commit body, "Out of nmsx Phase 1 scope": *"the test explicitly excludes
  these from the parity assertion with a documented exclusion list."*

**Why it's a concern:** the user said this verbatim: *"Do not let it create
backwards compatability bloat, or hacks to punt work for later."* The agent
found a real defect (mfmaIter deepcopy produces ids that don't match the
build_idmap inversion), did not fix it, added a test-level exclusion to
skip the failing case, and flagged it for "a future-bead concern."

The argument for the punt is "CMS-side `expand_cms_macro` also doesn't
re-tag mfmaIter sub-leaves." Examining ScheduleCapture.py:2486-2499
confirms: CMS's `expand_cms_macro` falls back to `mfma_classes` isinstance
→ MFMA, or UNKNOWN. CMS does NOT produce CVT_PACK/MIDDLE_PACK/LR for these
leaves; it produces MFMA/UNKNOWN. So the punt's premise "both sides have
the same gap" is **not** quite right: CMS's gap produces MFMA/UNKNOWN,
SHADOW's gap produces CVT_PACK/MIDDLE_PACK/LR. These are different
mismatches. The exclusion list excludes both sets of names from parity,
which papers over both gaps.

The "BPG#11 build advances past the SHADOW capture stage" claim in the
commit message is true only because the fail-loud raise is gated behind the
registry-recognized-but-no-canonical-tag fallback. Without that fallback,
the build would raise at the first deepcopied mfmaIter sub-leaf — which is
exactly what the design's fail-loud contract was designed to do (surface
producer-site / scope bugs immediately).

**Recommended fix:** the principled option is to either (a) drop the
deepcopy and find a way to share leaf id() into the SHADOW capture (so
build_idmap inversion catches them under their real PackA{u}/PackB{u}
category), or (b) acknowledge the deepcopy creates new ids and explicitly
re-tag via a leaf-walk after the deepcopy. Either way: the categories
CVT_PACK/MIDDLE_PACK/LR/LW/GR should not exist on SHADOW side, and the
exclusion list shouldn't need them.

---

### Concern 3: Per-subiter pack parity test asserts only set membership, not count equality
**Severity:** MEDIUM

**Evidence:**
- test_capture_pipeline_checks.py:330-338: the test computes
  `cms_pack_cats - shadow_pack_cats` (sets of category NAMES, not counts)
  and asserts the difference is empty. If CMS has 40 instances of `PackA0`
  and SHADOW has 1 instance of `PackA0`, this test passes.
- Fix 2's stated goal (commit body): SHADOW must capture `PackA1=40`,
  `PackB1=50` — counts, not just presence.

**Why it's a concern:** Fix 2 is verified, in spirit, only by
`test_shadow_main_capture_categories_match_cms_subject` (line 363+) which
DOES do count parity but excludes anything not in CMS's idMap. If
PackA{u}/PackB{u} categories are present in both but with different
counts, this would catch it — good. But the explicit per-subiter pack
coverage test is weaker than its docstring claims. Combined with the
exclusion list in Concern 2, the test surface for Fix 2 is thinner than
the commit message suggests.

**Recommended fix:** strengthen `test_shadow_main_capture_contains_per_subiter_packs` to also assert per-category COUNT equality for the
PackA{u}/PackB{u} keys it iterates. Two-line change.

---

### Concern 4: Sparse-MX metadata "defensively classified under LRSB"
**Severity:** MEDIUM

**Evidence:**
- KernelWriter.py:4807-4818 (mainloop): metadata leaves are put into
  `pointer_lrs_leaf_ids_B` with comment *"Metadata side: classify under
  LRSB defensively (sparse-MX metadata pairs with B in the CMS schema);
  fixture coverage (Phase 3) will surface a real case if this convention
  is wrong, at which point we'd add a per-side metadata bucket."*
- KernelWriter.py:3825 (NLL/NGL): same defensive classification.
- ScheduleCapture.py:1062-1065: build_idmap's metadata-aware path produces
  `LRMetadata{u}` (NOT LRSB) for sparse-MX metadata leaves. So the
  classification chosen here disagrees with the canonical idmap schema for
  sparse-MX builds.

**Why it's a concern:** this is a guess. The agent admits it ("defensively
classified"). Per design v5 §4 Phase 1 fail-loud contract, when in doubt
the leaf should raise so a fixture surfaces the real schema, not be
silently misclassified. Classifying metadata under LRSB also produces
SHADOW-side LRSB counts that include metadata in non-sparse builds — even
if the BPG#11 fixture has no metadata, future sparse-MX kernels (the very
fixtures Phase 3 enumeration is supposed to add per design v5 §4 Phase 3
fixture-coverage requirement) will see incorrect categorization.

**Recommended fix:** either (a) raise via the fail-loud contract when a
metadata-side leaf is encountered without an explicit per-side bucket
(`LRMetadataA`/`LRMetadataB`), or (b) plumb a proper
`pointer_lrs_leaf_ids_Metadata` set through the per-side map and have the
fix3 walk in `_makeSubIterSchedule` use the canonical `LRMetadata{u}`
category that matches `build_idmap`. The current "we'll find out in Phase 3
if it's wrong" is exactly the punt pattern.

---

### Concern 5: Root-level `tensilelite/conftest.py` mutating an editable-install finder is session-local plumbing that does not belong on a long-running branch
**Severity:** HIGH (for branch hygiene; does not affect runtime correctness)

**Evidence:**
- Verified `/home/alvasile/venv/lib/python3.11/site-packages/__editable___tensile_5_0_0_finder.py` MAPPING currently points at
  `/home/alvasile/rocm-libraries/.worktrees/validator_long_term_plans/projects/hipblaslt/tensilelite/Tensile` — a different worktree.
- The new `tensilelite/conftest.py` (lines 30-50) mutates that finder's
  in-memory `MAPPING` and `NAMESPACES` and (line 56-58) deletes preloaded
  `Tensile.*` modules from `sys.modules`.
- `git log --all -- "**/conftest.py"` confirms this file did not exist
  before commit 418a05c96e3.

**Why it's a concern:** this file solves a problem that exists only on
ONE machine (the agent's dev machine, where a venv has an editable
install pointing at a wrong path). Anyone else running these tests has a
clean venv where this conftest does nothing — fine in the happy path,
but `try: import __editable___tensile_5_0_0_finder ... except Exception:
pass` will silently swallow ANY import error including ones that indicate
a real test-collection bug.

More importantly: a long-running branch should not carry per-developer
session plumbing. The fix for "my editable install points at the wrong
worktree" is to `pip uninstall tensile && pip install -e .` in this
worktree, not to commit code that patches the install at import time.

**Recommended fix:** delete `tensilelite/conftest.py`. If tests fail with
import errors after deletion, fix by re-installing Tensile editable from
this worktree — not by re-committing a patch script.

---

### Concern 6: `fail_loud_on_missing_category=False` default preserves a code path scheduled for Phase 4 deletion
**Severity:** MEDIUM (backwards-compat bloat, by definition)

**Evidence:**
- KernelWriter.py:989 (`_makeSubIterSchedule`): `capture_fail_loud_on_missing_category=False`.
- KernelWriter.py:2712 (`_captureSubIterToBuilder`): same default.
- Approach-A call site at KernelWriter.py:4981-4998 does NOT pass the kwarg.
  Falls through to silent-UNKNOWN.
- Design v5 §4 Phase 4 (DEFAULT_SCHEDULER_REFERENCE_DESIGN.md line 157-164):
  Approach-A's `_captureNonCmsBuild` block and `build_non_cms_reference`
  are slated for deletion.

**Why it's a concern:** the kwarg-with-default-False exists ONLY to preserve
silent-UNKNOWN on a code path that's slated to die. The user's instruction:
*"Do not let it create backwards compatability bloat."* This is textbook
backwards-compat bloat — a flag whose entire raison d'être is "don't break
the thing we're about to delete."

Counter-argument: making fail-loud the only behavior would break Approach-A
mid-transition; the four-phase plan is intentionally serial; deferring
the contract on Approach-A is consistent with Phase ordering. That
argument has merit. But the cleaner shape would be to flip the default to
True and pass `False` explicitly at the one Approach-A call site, with a
TODO referencing Phase 4 — that makes the silent path visibly load-bearing
on a known-doomed callsite rather than blanket-default-tolerant.

**Recommended fix:** flip the default to `True`; pass `False` explicitly at
the single Approach-A call site (KernelWriter.py:4981) with a TODO comment
"Phase 4 will delete this call site; until then we tolerate the silent
fallback here." Makes the bloat localized and bound to a deletion event.

---

### Concern 7: `capture_pointer_side_map` and `closeLoopMod` plumbing
**Severity:** LOW (clean; flagged for completeness only)

**Evidence:**
- `capture_pointer_side_map` is a simple `{"lrs_a", "lrs_b", "lws_a", "lws_b"} -> set[int]` shape (KernelWriter.py:1058-1064, used at 1067-1083).
- `closeLoopMod` is computed once (KernelWriter.py:5169-5171) and passed to
  both the leftover-walk's `build_idmap` (line 5208) and
  `customMainLoopSchedule` (line 5304).
- Original `closeLoop()` invocation at line 5304 is now conditional on
  whether closeLoopMod was pre-computed.

**Why it's a concern (mild):** the per-side bookkeeping is a bit verbose
(every swap-offsets call site has a 3-line stanza for the leaf-id walk)
but the alternative (a wrapping helper) would obscure the per-side
determination at the call site. The pattern is acceptable. The
`closeLoopMod` change does add a contract that "passing the same module
twice is fine" — verified at KernelWriter.py:5294 + 5304; closeLoop is
stateless w.r.t. the returned Module (it generates a new tree each
invocation, so passing the same instance twice can only result in the
SAME mutable instance being shared; if either consumer mutates the module
in place, the other is affected). Spot-check of
`customMainLoopSchedule` for in-place mutation of its `loopCode` arg is
out of scope for this review but the agent's commit message claims "no
observable side effects" — accept on trust unless Phase 3 surfaces a
defect.

**Recommended fix:** none. This is fine.

## Plan-adherence: matches plan

- Fix 1 (LCC) lands at the correct site (the end-of-loop SHADOW finalize)
  using the design-mandated approach: harvest into the SHADOW builder via
  `_appendCloseLoopLCCToBuilder` before `builder.finalize()`. Same
  closeLoopMod instance is reused for `customMainLoopSchedule` to avoid
  double-emission of side effects. KernelWriter.py:5169-5171, 5294-5295.
- Fix 2 (PLR1 packs) restores the leftover walk over `pack[*]/packPre[*]`
  with `num_loop_iter=len(LRCodeAAllIters)` and includes the xbi0
  (same-id) and flpk (canonical-text) invariants documented in
  KernelWriter.py:5237-5265.
- Fix 3 (LRS/LWS schema) does the per-side tagging at the producer
  (production call sites) rather than parsing text downstream — the
  principled location per design v5 §4 Phase 1 Fix 3 statement.
- `CaptureCategoryMissingError` is defined in ScheduleCapture.py:117-139
  per design.
- Unit tests added.
- Tests pass: 27 passed in 5.09s (verified).

## Plan-adherence: deviates from plan

- **Graceful fallback for CVT_PACK/MIDDLE_PACK** (Concern 1). Deviation
  from design v5 §4 Phase 1 fail-loud contract. **NOT ACCEPTABLE** as-is
  per the user's "no hacks to punt work for later" rule. The fallback
  produces tags that provably cannot match CMS-side schema, so it
  captures bytes but loses the comparison contract.
- **mfmaIter deepcopy left in place + test exclusion list** (Concern 2).
  Deviation from design v5 §4 Phase 1 acceptance ("SHADOW capture
  per-category counts on BPG#11 match CMS subject's modulo SYNC/SNOP").
  **NOT ACCEPTABLE** as-is per the user's "no hacks to punt work for
  later" rule.
- **Sparse-MX metadata defensively classified as LRSB** (Concern 4).
  Deviation from design v5 §4 Phase 1 fail-loud contract. **NOT
  ACCEPTABLE** as-is — the design says "raise when uncertain," not "guess
  with a TODO."
- **Root-level conftest.py** (Concern 5). NOT in plan. **NOT ACCEPTABLE**
  — local dev plumbing belongs in dev's gitignored space, not on a
  long-running branch.
- **`fail_loud_on_missing_category` default-False kwarg** (Concern 6).
  Defensible deviation (Approach-A is Phase 4 retirement scope), but the
  current shape inverts the relationship between principled behavior and
  legacy path. **ACCEPTABLE IF REFACTORED** to flip the default with an
  explicit single-site opt-out for the doomed Approach-A call.

## Tests

- 27 tests pass in 5.09s. Verified.
- The 7 new tests do exercise the contracts as described, except for
  Concern 3 (count-vs-set assertion gap) and Concern 2 (exclusion list
  masking the punted case).
- `test_synthetic_unregistered_class_raises` (line 451) is well-formed:
  uses VXorB32 (verified absent from `_CLASS_NAME_TO_CATEGORY`), feeds
  through the real `_captureSubIterToBuilder`, asserts on message
  contents including "DEFAULT_SCHEDULER_REFERENCE_DESIGN". Good.
- `test_synthetic_unregistered_class_silent_when_fail_loud_off` (line 499)
  is the canary for Approach-A's silent-UNKNOWN path — load-bearing per
  Concern 6 above. If Concern 6 is addressed by flipping the default and
  passing False explicitly, this test should be updated to test the
  Approach-A call site's explicit-False rather than the function default.
- `test_shadow_main_capture_categories_match_cms_subject` (line 363) is
  a good parity test in shape, but the exclusion list (line 398-399)
  removes most of the categories where Concerns 1+2's defects would
  manifest. The remaining covered categories (LRA{u}/LRB{u}/LRSA/LRSB/
  LWA/LWB/LWSA/LWSB/PackA{u}/PackB{u}/LCC/GRIncA/GRIncB) do exercise the
  three fixes — but only because Concerns 1+2 ensure the failing
  categories never reach this assertion.

## Final recommendation

Next agent must address these in priority order:

1. **(BLOCKING)** Fix Concern 1: remove the `inst_cat.name` graceful
   fallback at KernelWriter.py:2924-2926. Either raise (strict design
   reading) or extend Fix 2's per-iter PackA{u}/PackB{u} coverage to the
   leaves that currently trigger this path.

2. **(BLOCKING)** Fix Concern 2: either share leaf ids across the
   mfmaIter deepcopy (eliminating the new-id problem) or do a post-deepcopy
   re-tag walk that maps new-id sub-leaves into their PackA{u}/PackB{u}
   category. Then remove `CVT_PACK`, `MIDDLE_PACK`, `LR`, `LW`, `GR` from
   the parity-test exclusion list (test_capture_pipeline_checks.py:398-399).

3. **(BLOCKING)** Fix Concern 5: delete `tensilelite/conftest.py`. Fix
   any resulting test-collection import errors by re-installing the
   editable Tensile from this worktree, not by re-introducing a patch
   script.

4. **(SHOULD)** Fix Concern 4: handle sparse-MX metadata via the
   canonical `LRMetadata{u}` category from `build_idmap`, or raise per
   the fail-loud contract. Don't guess.

5. **(SHOULD)** Fix Concern 6: flip `fail_loud_on_missing_category`
   default to True; pass False explicitly at the one Approach-A call site
   with a Phase 4 TODO.

6. **(NICE-TO-HAVE)** Fix Concern 3: strengthen
   `test_shadow_main_capture_contains_per_subiter_packs` to assert
   per-category count equality, not just set coverage.

After 1+2 land, re-run the parity test with the exclusion list trimmed
to design-legitimate exclusions only (SYNC/SNOP/SSETPRIO/SBARRIER). If
that passes on BPG#11, the design v5 §4 Phase 1 acceptance criterion is
genuinely met. The current state passes a weaker test that masks the
exact failures the contract was written to expose.
