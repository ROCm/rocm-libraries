# e293 R1 — Approach 2 vs Approach 3 Comparison Memo

Author: comparison agent (Claude Opus 4.7, 2026-05-12). **Investigation
only — no production code changes.** Inputs: `/tmp/E293_R1_APPROACH_2_MEMO.md`
(29240 B), `/tmp/E293_R1_APPROACH_3_MEMO.md` (28063 B), reviewer
verdicts (Approach 2: APPROVE WITH MINOR CAVEAT; Approach 3: APPROVE),
HDEM_IMPLEMENTATION.md §"Implementation surprises" §1.

---

## 1. Executive summary

Both approaches are reviewer-approved and close the 12 e293 SCC false
positives cleanly using rocisa-derived information. Approach 2 adds a
third slot (`source_module_id`) to identity itself — wider, more
invasive, with stronger structural guarantees but six test files to
migrate. Approach 3 adds a scoped suppression gate inside
`diagnose_missing_edge` that consults `node.category` as a tiebreaker
on aliased-singleton resources — narrower, localized, lower test
churn, but a residual silent-suppression failure mode if its
positive-evidence predicate ever misfires.

---

## 2. Side-by-side comparison table

| Axis | Approach 2 (identity slot) | Approach 3 (comparator gate) |
|---|---|---|
| **1. e293 fix correctness** | Closes all 12 SCC false positives by giving GRIncA-cmp and GRIncB-cmp distinct identities. Mechanism is structural (no chance of miss). | Closes all 12 by suppressing the OverriddenInputFailure + Phase-1 OrderInverted emissions when the cross-build identity-collision fingerprint is positively confirmed. Mechanism is heuristic-by-construction but predicate (4) demands positive evidence. |
| **2. Coverage envelope** | Fixes all "same canonical_render, different source module" cross-build collisions: e293 SCC, future m0 GRA-vs-GRB, and any new cross-side same-render twin. Does NOT fix p39d GRA/GRB OrderInverted (wrong mechanism — per-tile schedule), ldm5 missing MFMAs (pre-comparator), UsePLRPack cross-body movement, or wait-coverage residuals. | Fixes e293 SCC and future m0 collisions explicitly. Does NOT fix p39d, ldm5, UsePLRPack, or wait-coverage residuals (per memo §Q5 — same exclusions). Same-render same-side collisions not addressed (predicate (2) fails). |
| **3. xqj3 alignment** | Adds a rocisa-derived slot (Module.name from rocisa-side constructor calls). Does NOT re-introduce CMS-shaped strings into identity. xqj3-clean. | Identity layer untouched. Comparator consults `node.category` (CMS-shaped) as a tiebreaker, scoped to a new `_SIDE_DISCRIMINATING_CATEGORIES = {"GRIncA","GRIncB"}` enum. Pattern is structurally identical to existing `counter_for(node)` and `_RECOGNIZED_CATEGORY_PREFIXES` consumers. xqj3-acceptable but stretches the line: identity-clean, comparator borderline. |
| **4. Implementation cost** | ~200–250 LOC; touches 5 files (ScheduleCapture, KernelWriter, CustomSchedule/dispatch, CMSValidator no-logic verification, 6 test files). Plumbing depth: 3 layers (idmap → builder → identity). Test migration: 6 identity-shape pinning files. | ~75–90 LOC production + ~150 LOC tests; touches 1 file (CMSValidator). Plumbing depth: 0 (gate inserted at two existing call sites). Test migration: zero existing identity tests touched; 7 new tests. |
| **5. Risk surface** | Failure if a future kernel-writer refactor renames `globalReadIncrementA` → `gri_A` (identity becomes intra-version-only, but already true of `canonical_render`). Risk if Module-name uniqueness assumption breaks (mitigated via positional-fallback). Risk if unnamed intermediate Modules confuse the closest-named-ancestor walk. Failures are LOUD: identity-set mismatch is visible. | Risk A (largest): silent suppression of a real same-render same-side defect that happens to satisfy all four predicates. Mitigated by predicate (4) requiring intact peer chain in subj. Risk B: gate fails to fire when peer chain has legitimate intervening dataflow-irrelevant ops; needs adjacency tolerance. Risk D: depends on `node.category` correctness from capture pipeline. Failures are SILENT: most painful failure mode for a validator. |
| **6. Reversibility** | Hard. Identity-tuple shape is consumed by 6 test files plus opaque consumers in `nodes_by_identity` / `edge_keys`. Rolling back means reverting the migration across all of them. NamedTuple migration (if taken) further entrenches the slot. | Easy. Two call-site insertions plus one helper module — `git revert` cleanly. The new `_SIDE_DISCRIMINATING_CATEGORIES` enum is unreferenced elsewhere. |
| **7. Composability** | Does NOT preclude landing Approach 3 later as a defense-in-depth gate (the gate becomes a no-op once identity disambiguates). | Does NOT preclude landing Approach 2 later (the identity refactor would obviate the gate, which can then be removed). The gate is a strictly weaker, additive intervention. |
| **8. Surprise-vs-spec match** | Changes IDENTITY — visible in every failure message, every pin test, every printed identity tuple. Structurally invasive. Matches "make the structural problem hard to recreate": once identity carries source-module discrimination, future engineers cannot accidentally re-create the cross-build aliasing. | Changes COMPARATOR LOGIC — invisible outside `diagnose_missing_edge`. Localized. Does NOT make the structural problem hard to recreate: a future singleton-resource family beyond SCC/m0 (or a third side beyond A/B) requires explicit gate extension. The structural defect (same canonical_render aliases on cross-build identity) PERSISTS in identity; the gate just papers over its observable consequences. |

---

## 3. Per-axis discussion

### Axis 1 — e293 fix correctness

Approach 2 fixes by construction: per Approach 2 §Q4 proof sketch
steps 1-5, GRIncA-cmp and GRIncB-cmp inherit `source_module_id`
`"globalReadIncrementA"` and `"globalReadIncrementB"` from the
unconditionally-executed `Module(...)` constructor calls at
`KernelWriterAssembly.py:9170/9178`. The cross-build identity-pair
collision cannot recur for these two emissions. Approach 3 fixes by
positive-evidence detection: per Approach 3 §Q3, the gate confirms
the mismatched `p_node.category` vs `ref_edge.producer.category` AND
that the TRUE peer's chain is intact in subj before suppressing. Both
close all 12 failures across the 3 fixtures. Approach 2's correctness
is structural; Approach 3's is by case-analysis with a documented
positive-evidence discipline (predicate 4).

**Reviewer caveat on Approach 2's coverage statement** (`_loopBody`'s
`UseCustomMainLoopSchedule` branches mean LR/Pack source modules ARE
asymmetrically populated): I agree with the reviewer. Approach 2 §Q5
correctly scopes the cross-build stability claim to source modules
constructed by the SHARED kernel-writer code path (GRIncA/B, GR-side
m0). LR/Pack modules are populated differently in the CMS branch
(`KernelWriter.py:4671-4687`) vs the non-CMS branch (`:4591-4607`)
because some intermediate Modules ARE constructed inside CMS-only
branches. The proof in Approach 2 §Q4 step 1 must be sharpened: it
holds for the cited e293 case (GRIncA/B) but does not generalize to
ALL source modules. This sharpening does not invalidate the e293 fix —
GRIncA/B construction IS unconditional — but it narrows the implicit
"all rocisa Modules" coverage claim.

### Axis 2 — Coverage envelope

Approach 2 fixes the broader class "same canonical_render, different
source module, cross-build". This subsumes e293 SCC + the GR-side m0
twins (`s_add_u32 m0, m0, 4224` from GRA vs GRB, called out in
`EMISSION_ORDINAL_DESIGN.md §1.2`) + any new cross-side same-render
twin. Approach 3 fixes e293 SCC + m0 explicitly, plus any other
"aliased singleton resource + side-discriminating category" collision —
its scope is parameterized by `_is_aliased_singleton_resource()` AND
`_SIDE_DISCRIMINATING_CATEGORIES`. Concretely: Approach 2 fixes a
single SUPERCLASS (cross-build same-render aliasing) once;
Approach 3 fixes one bounded family (SCC+m0 on GRIncA/GRIncB)
explicitly and requires a code change for each new singleton family.
Neither fixes p39d (per-tile schedule), ldm5 (pre-comparator),
UsePLRPack (cross-body movement), or wait-coverage residuals — these
are orthogonal mechanisms.

### Axis 3 — xqj3 alignment

Both claim rocisa-derivation. Verifying depth:

* **Approach 2.** `Module.name` is set in rocisa-side constructor
  calls at `KernelWriterAssembly.py:9170/9178` (the kernel writer
  *is* the rocisa-derived layer; CMS scheduling reorders within
  `Module.flatitems()` but never renames Modules). The new identity
  slot is populated from rocisa-side state, never from a CMS-shaped
  string. The claim is rigorous. The residual coupling is to chosen
  string names; this is the same shape of coupling `canonical_render`
  already has to chosen instruction-render text — not a new
  CMS-coupling.
* **Approach 3.** Identity layer remains untouched (per memo §Q7
  table — `identity_for`, `_role`, `_DATA_FLOW_CATEGORIES`
  unchanged). The comparator gate consumes `node.category` (CMS-
  shaped) as a tiebreaker. The memo correctly notes the precedent
  (`counter_for(node)`, `_RECOGNIZED_CATEGORY_PREFIXES`,
  `_split_category_iter` all consume `node.category` for bounded
  comparator purposes). The xqj3 line is "identity must be rocisa-
  derived"; Approach 3 holds that line. The honest read: Approach 3
  is xqj3-acceptable but does not advance xqj3's broader goal of
  removing CMS-string surface from the validator. Approach 2 advances
  xqj3 by adding a rocisa-derived slot.

### Axis 4 — Implementation cost

Approach 2: ~200–250 LOC across 5 files (ScheduleCapture,
KernelWriter, CustomSchedule/dispatch, CMSValidator verification, 6
test files). Plumbing threads `source_module_id` through 3 layers
(idmap inversion → builder.append → identity_for). Comparable in
scope to 4up4 / hdem migrations per memo §3 estimate. Approach 3:
~75–90 LOC production + ~150 LOC tests, all in CMSValidator.py.
Plumbing depth: 0 (consumes pre-existing `node.category`). Sizing
is a sizing input, not a recommendation lever — but the scope
difference is real and informs reversibility (axis 6).

### Axis 5 — Risk surface

This axis differentiates the approaches sharpest. **Approach 2's
risks are LOUD:** if Module-name uniqueness breaks, identity-set
mismatch surfaces immediately. If a future refactor renames a Module,
identity stability across versions breaks visibly (and is intra-
version anyway). If unnamed intermediate Modules confuse the
ancestor walk, the `source_module_id` falls back to the positional
enumeration — degrading discrimination but not introducing silent
errors. **Approach 3's largest risk (Risk A) is SILENT:** if a real
same-render same-side defect coincidentally satisfies all four
predicates, the gate suppresses it. The mitigation (predicate 4
requires intact peer chain in subj) narrows the conjunction but
cannot eliminate it. For a validator, silent absorption is the most
painful failure mode — the bug under detection is the absence of a
bug report. Approach 3's memo §"Risk A" honestly acknowledges this
and identifies predicate (4) as the principled (non-relaxable)
mitigation.

### Axis 6 — Reversibility

Approach 2 is hard to reverse: 6 test files migrated, 5 production
files touched, NamedTuple migration (if taken) entrenches the
identity shape. Rolling back means a counter-migration. Approach 3
is easy to reverse: `git revert` the single PR; the new
`_SIDE_DISCRIMINATING_CATEGORIES` enum is unreferenced elsewhere; no
other consumers depend on the gate's existence.

### Axis 7 — Composability

Neither precludes the other. If Approach 3 lands first and proves
insufficient (e.g., a third side-discriminating category emerges),
Approach 2 can land on top — once identity disambiguates, the gate
becomes a confirmed no-op and can be removed. If Approach 2 lands
first, Approach 3 becomes unnecessary because the structural collision
no longer occurs. Approach 3 is therefore a strictly additive
defense-in-depth that does not foreclose Approach 2's structural fix.

### Axis 8 — Surprise-vs-spec match

This is the axis that the user's stated principle ("make the
structural problem hard to recreate") most directly bears on.
Approach 2 makes the structural defect hard to recreate: any future
kernel-writer engineer who introduces a third source module emitting
the same canonical render will immediately see the new module's
`source_module_id` propagate into identity tuples and disambiguate
naturally. The fix is in the abstraction. Approach 3 makes the
SYMPTOM hard to recreate: the gate suppresses the observable
consequences of the cross-build collision, but the collision itself
remains in identity. A future engineer adding a third aliased
singleton resource (or extending the side-discriminating set) must
modify `_is_aliased_singleton_resource` and
`_SIDE_DISCRIMINATING_CATEGORIES` explicitly. This is the
"principled-for-the-operative-case, NOT principled-in-full-generality"
shape — Approach 3 is the localized lighter mechanism that requires
ongoing case-by-case maintenance.

---

## 4. Historical precedent — hdem byte-key vs identity decision

`HDEM_IMPLEMENTATION.md §"Implementation surprises" §1`: the literal
byte-key proposal in ORAM1 §4 lost producer-discrimination on real
test cases (`swap_pack`, `LR_first/LR_second`). The hdem implementer
chose identity-based matching — the more rigorous, structurally-wider
fix — because the lighter byte-key proposal silently collapsed
distinct producers into one. The reviewer in that round noted the
choice was "principled-for-the-operative-case, NOT principled-in-
full-generality" for the LIGHTER mechanism (byte-key). The user
accepted the heavier identity-tuple mechanism for the soundness
guarantee.

Approach 2 vs Approach 3 has the same shape, but the rigor mapping
is opposite to a naive read: in this case, **Approach 2 is the
heavier identity-level fix (analogous to hdem's chosen identity-
based matching), and Approach 3 is the lighter localized fix
(analogous to the byte-key proposal that hdem rejected).** Both
hdem decisions and the e293 decision turn on the same question: is
silent absorption acceptable in exchange for lower mechanism cost?
The hdem precedent answered NO. If the e293 R1 decision applies the
same standard, Approach 2 wins on the soundness axis; if the user
weights reversibility / scope-narrowness more heavily this time,
Approach 3 is the rational pick.

The honest distinction: hdem's byte-key proposal failed on PRESENT
real test cases (`swap_pack`). Approach 3's silent-absorption risk
is HYPOTHETICAL (predicate 4 narrows the conjunction; no current
fixture exercises it). The hdem precedent is therefore directionally
informative but not dispositive.

---

## 5. Recommendation

**Recommended: Approach 2 (identity-slot fix), conditional on the
reviewer caveat being addressed.**

**Dominant axis: Surprise-vs-spec match (axis 8) + risk-failure-mode
shape (axis 5).**

The user's stated principle is "make the structural problem hard to
recreate." Approach 2 satisfies this directly by closing the
structural defect in identity. Approach 3 papers over the symptom and
leaves the defect in place. The hdem precedent (axis 4) reinforces
this: when a lighter localized fix has a silent-failure mode and a
heavier structural fix exists, the heavier fix is the principled
choice. Approach 3's silent-absorption Risk A — a real same-render
same-side defect coincidentally satisfying all four predicates — is
narrow but cannot be eliminated, and silent absorption is the most
painful failure mode for a validator. Approach 2's failures are loud:
identity-set mismatches surface immediately.

**Sub-condition Y: Approach 2 should land only after the §Q5
coverage statement is sharpened to scope the cross-build stability
claim to source modules constructed by the SHARED kernel-writer code
path** (GRIncA/B + GR-side m0 + GRA/GRB family) **rather than
implying ALL rocisa Modules are covered.** The reviewer caveat on
LR/Pack source-module asymmetric population must be reflected in the
implementation memo's scope statement before code lands. The fix
itself remains correct for the e293 case; the scope claim needs
narrowing.

Counter-considerations honestly acknowledged:

* **If reversibility is weighted higher than soundness** (e.g., the
  user wants to ship a fix that can be cleanly backed out if it
  destabilizes the test pipeline), Approach 3 wins on axis 6.
* **If implementation cost is a hard constraint** (e.g., 4up4/hdem
  migration fatigue), Approach 3 wins on axis 4 — but the user's
  stated guidance ("anti-language ... 'X LoC' as recommendation
  lever prohibited") rules out making this dominant.
* **If the user accepts silent-absorption risk as cost-of-doing-
  business** for narrower-scope fixes (matching the hdem reviewer's
  "principled-for-the-operative-case" framing applied differently),
  Approach 3 is rational. The hdem decision did NOT accept that
  trade-off; it chose the heavier mechanism.

A defense-in-depth composition (Approach 2 lands first, Approach 3
becomes a confirmed no-op and is removed) is mechanically supported
by axis 7 — but is not recommended as the initial play because it
doubles implementation cost without meaningful soundness gain.

---

## 6. Open questions for the user

1. **Does the user weight soundness (no silent absorption) over
   reversibility / localized scope for this specific fix?** The
   recommendation hinges on this. If the answer is "yes, soundness
   first" (consistent with the hdem precedent), Approach 2 is the
   pick. If "this time prioritize narrow blast radius," Approach 3
   is the pick. Both are reviewer-approved; neither is wrong.

2. **Is the §Q5 coverage sharpening (sub-condition Y) acceptable as
   a pre-implementation memo edit, or does it require a re-investigation
   of LR/Pack source-module construction asymmetry?** The reviewer's
   note suggests the design itself is sound but the coverage CLAIM
   over-generalizes. A memo edit should suffice, but the user may
   want explicit verification that the GR-side m0 case (the second
   advertised coverage benefit) survives the same scoping discipline.

Additional clarifying questions surfaced by the comparison but not
required for the pick:

3. Should the identity-slot migration (Approach 2) take the
   opportunity to migrate identity from a bare tuple to a NamedTuple
   for migration resilience? (Per Approach 2 §5 Q2 — out of scope of
   the e293 fix proper but a natural moment.)
4. If Approach 3 is chosen instead, should `_has_intact_carry_chain`
   adjacency tolerance default to "skip
   `_NO_DATAFLOW_IDENTITY_CATEGORIES`" or strict "next aliased-
   singleton writer in body"? (Per Approach 3 §5 Q1.)

---

## 7. Citations

* `/tmp/E293_R1_APPROACH_2_MEMO.md` — Approach 2 source memo (29240 B).
* `/tmp/E293_R1_APPROACH_3_MEMO.md` — Approach 3 source memo (28063 B).
* Reviewer verdicts (verbatim, summarized in §1 / §3).
* `Tensile/Components/HDEM_IMPLEMENTATION.md` §"Implementation
  surprises" §1 — byte-key-vs-identity precedent (read at lines
  277-306 of the validator_long_term_plans worktree copy).
* `Tensile/Components/E293_SCC_CARRY_CHAIN_INVESTIGATION.md` —
  mechanism trace (cited transitively via both source memos).
