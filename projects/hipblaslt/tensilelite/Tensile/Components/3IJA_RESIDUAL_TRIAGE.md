# 3ija: Triage compare_graphs / wait-coverage residuals on real Build #2 across the gfx950 CMS test surface

Bead: `rocm-libraries-3ija` (P0, sub-bead of `rocm-libraries-71hw`).
Investigator: 3ija-investigator (Claude Opus 4.7, May 2026).
Working tree: `/home/alvasile/rocm-libraries/.worktrees/validator_long_term_plans/`.
Base branch: `users/alvasile/validator_long_term_plans` (tip `59bb5a8e9f` — nyb5).

**Investigation primary, no production-code fixes inline.** Output is this
memo + sub-bead filings for each residual class.

## Q2 framing (recurring)

Per `2LZD_INVESTIGATION.md §6.2 Q2`:

> Validation comparison is YAML-in, two builds out (one with
> `UseCustomMainLoopSchedule=1`, one with `=0`). Whatever Tensilelite
> mutates internally on either side is accepted as part of "what that
> build does."

Each surfaced residual is classified per the bead description:
- **(a) Real defect** — Tensilelite produces divergent code that should agree. Fix at source.
- **(b) Expected scheduler difference per Q2** — accept and exclude from comparison.
- **(c) Validator logic divergence** — `compare_graphs` treats the same input differently on the two sides for legitimate reasons. Fix at the validator.

Discipline rule #8 (from the bead description): any residual that does
not classify cleanly into (a)/(b)/(c) is surfaced as a fourth category
needing user input.

---

## §1 — CMS test surface exercised

### Triage runner

The investigation's primary tool is
`Tensile/Tests/unit/_3ija_residual_triage_runner.py` — a pytest-driven
investigation script that:

1. Imports the `Tensile.Components.CustomSchedule.gfx950` package
   (which side-effect-loads every `RegisterSchedule` decorator).
2. Iterates `_SCHEDULE_METADATA` for every (name, layout, LDSTr,
   TLDS) combo with `dtype in {"TF32", "16bit"}` (8-bit out of scope
   per the 5gd rescope reference).
3. For each fixture: builds the kernel config from `CMSKernelInfo`,
   calls `_make_solution`, runs Build #1 (CMS) via
   `KernelWriterAssembly._getKernelSource`, runs Build #2 via
   `build_non_cms_reference`, then runs
   `compare_graphs(ref=Build#2, subj=Build#1)` and
   `validate_edge_wait_coverage(ref_graph)`.
4. Classifies each Failure by `(type, prod_cat, cons_cat, prod_body, cons_body)`.

Run with:
```
pytest Tensile/Tests/unit/_3ija_residual_triage_runner.py -s \
    --ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py
```

### Fixture coverage

The dispatch registry produced **93 (fixture, layout, LDSTr) combos**
across 37 schedule files (16 TF32 + 21 16-bit). Of those:

| Outcome | Count | Notes |
|---|---|---|
| Cleanly exercised, 0 residuals | 27 | Mostly 16-bit TN fixtures — Approach A passes cleanly today |
| Exercised, non-zero residuals | 6 | The (a)/(b)/(c)-classifiable surface |
| `_make_solution` rejected (not valid for config) | 36 | **Tooling artifact** — schedule registry's auto-detection probe over-reports layouts that the actual half/bf16/TF32 Solution validator rejects |
| `_last_cms_capture is None` (CMS Build #1 didn't populate capture) | 6 | 3 fixtures × 2 LDSTr variants — see §3.E |
| `compare_graphs` raised `CaptureUnknownInstructionError` | 12 | Classifier-registry gap — see §3.C |
| `compare_graphs` raised `UnexplainedMissingEdgeError` | 4 | Classifier-dispatch gap — see §3.D |
| `compare_graphs` raised `CaptureConsistencyError` (MFMA missing in subject) | 2 | Structural divergence — see §3.F |

The **36 `_make_solution` rejections** are NOT 3ija-scope residuals.
The schedule registry's `_detect_supported_layouts` probe at
`dispatch.py:552` uses a `_ProbeDataType` stub (returns
`numBytes()=2`, no real `isHalf()`/`isBFloat16()` etc.) that accepts
configs the actual Solution validator rejects. Those rejections fire
with messages like `"reject: UseCustomMainLoopSchedule=1 but CMS is
not supported"` — i.e. the runtime Solution path doesn't think the
combo is CMS-compatible. This is a known gap in the auto-detection
probe (separate from this bead's scope).

The **6 fixtures with `cg=0 wc=0` AND `LDSTr=True`** indicate the
LDSTr-True variants behave identically to LDSTr-False on those
schedules. No new residual signal there.

---

## §2 — Per-fixture residual table

Showing only fixtures with non-zero compare_graphs residuals:

| Fixture | Layout | LDSTr | CG count | WC count | Failure shape (count × type, producer→consumer, body) |
|---|---|---|---|---|---|
| `_128x128x32_TF32` | TN | False | 3 | 0 | 2× OrderInverted GRA→GRA ML→ML; 1× OrderInverted GRB→GRA ML→ML |
| `_128x256x32_TF32` | TN | False | 1 | 0 | 1× OrderInverted GRB→GRB ML→ML |
| `_192x256x32_TF32` | NN | True | 5 | 0 | 2× OverriddenInput GRIncB→GRIncA NGL→NGL; 2× OrderInverted GRIncA→GRIncB NGL→NGL; 1× OrderInverted GRA→GRB ML→ML |
| `_192x256x32_TF32` | NN | False | 5 | 0 | (identical to LDSTr=True row above) |
| `_256x160x64_16bit` | TN | False | 4 | 0 | 2× OverriddenInput GRIncB→GRIncA NGL→NGL; 2× OrderInverted GRIncA→GRIncB NGL→NGL |
| `_256x256x32_TF32` | TN | False | 1 | 0 | 1× OrderInverted GRA→GRA ML→ML |

All wait-coverage residuals are **0 across every exercisable fixture**.
That's a clean result for `validate_edge_wait_coverage` on the
non-CMS reference graph: no missing wait, no insufficient wait, no
missing barrier, no middle-pack pair-interleaving violation.

### Aggregate compare_graphs residual shapes

Across the 6 affected fixtures (19 total CG residuals):

| Count | Shape |
|---|---|
| 5 | OrderInvertedFailure, GRA/GRB → GRA/GRB, body=ML |
| 4 | OrderInvertedFailure, GRIncA → GRIncB, body=NGL |
| 4 | OverriddenInputFailure, GRIncB → GRIncA, body=NGL (resource=SCC) |

Plus the error-case residuals (which prevent reaching
compare_graphs at all):

| Count | Shape |
|---|---|
| 12 | CaptureUnknownInstructionError on VSwapB32 / VPermB32 / VOrB32 / VMovB64 in body=ML-1 |
| 4 | UnexplainedMissingEdgeError on `ds_read_b128 → v_cvt_pk_bf16_f32` (LR → CVT) raw_intrawave |
| 6 | `_last_cms_capture is None` (CMS Build #1 didn't populate capture) |
| 2 | CaptureConsistencyError, 20-28 MFMA identities in ref but not subject |

---

## §3 — Residual classification with mechanism citations

### A. OrderInvertedFailure on GRA/GRB in ML (5 occurrences across 4 fixtures)

**Classification: (b) — expected scheduler difference per Q2.**

This is the same shape nyb5's Cycle 2 surfaced and pinned by
`test_non_cms_reference_compare_graphs_surfaces_only_known_residuals`
on the canonical TF32 4×4 TN config. The 3ija triage confirms it
**generalizes across the gfx950 CMS surface** — at least 4 distinct
schedule fixtures exhibit it.

Mechanism (per `NYB5_IMPLEMENTATION.md §"Cycle 2 surprise"`):

> The per-tile schedule on the CMS side mutates kernel-level flags
> (`UsePLRPack=True`, `UseMFMAF32XEmulation`) before SIA3 runs, which
> changes the GR scheduling order. … The non-CMS reference build
> (Approach A) uses an unmutated `kernel` dict per
> `2LZD_INVESTIGATION.md §6.2 Q2`. So the GR-stream divergence is the
> *expected* Q2 surfacing.

Source citation: per-tile schedules at e.g.
`Tensile/Components/CustomSchedule/gfx950/_128x128x32_TF32.py:119-120`
set `kernel["MfmaInitCVgprs"] = True` and `kernel["UsePLRPack"] = True`
inside the schedule registration body, mutating the CMS-side kernel
dict.

**Filed as:** `rocm-libraries-p39d` — proposes 3 candidate principled
fixes (extend `_NO_DATAFLOW_IDENTITY_CATEGORIES`, add per-edge
tolerance, or body-blind GR identity). **Not implemented inline** —
the bead description allows a one-line addition to
`_NO_DATAFLOW_IDENTITY_CATEGORIES` only when "the fix is clearly
accept-this-difference rather than mask-this-difference." For GR
ordering, none of the three options is unambiguously the right
mechanism without design review (see §6 open question 2 below).

### B. GRInc OrderInverted + OverriddenInputFailure(SCC) pair in NGL (4+4 occurrences across 3 fixtures)

**Classification: NEEDS USER INPUT — discipline rule #8 surface.**

Affected fixtures: `_192x256x32_TF32 NN` (both LDSTr variants),
`_256x160x64_16bit TN`.

The OrderInverted and OverriddenInputFailure fire on the same
producer/consumer pair in opposite directions, suggesting CMS
reordered GRIncA/GRIncB AND an intervening writer (likely SCC carry
from one GRInc) is between the producer and consumer of the other
GRInc. Resource is **SCC** (the ALU carry register).

This sits between (a) and (b):

- **(b)-leaning:** GRIncA/GRIncB are scheduler-chosen address
  increments; same Q2 mechanism as §3.A.
- **(a)-leaning:** OverriddenInputFailure on `resource=SCC` is
  structurally a real defect — SCC clobber means carry-chain
  arithmetic produced wrong addresses. Unlike GR ordering (just two
  memory reads in a different order), SCC clobbering breaks the
  address increment computation. If true, the SHADOW comparison
  hid this (worst-case shadow-defect: false green from shared
  state).

**Filed as:** `rocm-libraries-e293` (priority 0, type=question).
Recommends investigating the actual emitted asm before classifying.
This is the only residual class that explicitly invokes the discipline
rule #8 surface-to-user.

### C. CaptureUnknownInstructionError on VSwapB32 / VPermB32 / VOrB32 / VMovB64 in ML-1 (12 fixtures)

**Classification: (c) — validator logic gap.**

`Tensile/Components/CMSValidator.py:_make_node` (and ultimately the
rocisa-derived classifier registry per
`EMISSION_ORDINAL_DESIGN.md §4.1`) does not recognize these instruction
classes. The non-CMS reference build emits them as part of pack/swap
chains; the SHADOW-vs-CMS comparison masked this because the shadow
was a synthetic re-assembly that did not include these instructions.

Affected fixtures:
- VSwapB32 (5): `_256x256x64_16bit NN/NT/TT`, `_192x256x64_16bit NT`,
  `_256x256x32_TF32 NT`
- VPermB32 (4): `_128x128x32_TF32_plr1 NN`, `_128x128x64_TF32 NN`,
  `_128x160x64_TF32 TN`, `_160x128x64_TF32 TN`
- VOrB32 (2): `_96x256x64_16bit NN` and others
- VMovB64 (2): TF32 fixtures

**Filed as:** `rocm-libraries-6hk3` — proposes extending the
classifier registry. Not implemented inline (needs per-class
category audit; not a one-liner).

### D. UnexplainedMissingEdgeError on LR → CVT (`ds_read_b128 → v_cvt_pk_bf16_f32`) raw_intrawave (4 TF32 fixtures)

**Classification: (c) — validator logic gap.**

`compare_graphs`'s `diagnose_missing_edge` cannot classify an edge
where an LR feeds a CVT (TF32-emulation pack convert) directly. The
existing dispatch table has CVT_PACK and PACK_MFMA branches but no
LR → CVT entry. The TF32 emulation path (per `UseMFMAF32XEmulation`)
introduces these convert instructions on the non-CMS side; the CMS
side's per-tile schedule may relocate them.

Affected fixtures: `_64x128x64_TF32 TN`, `_128x64x64_TF32 TN`,
`_128x128x32_TF32_plr1 TN`, `_128x128x64_TF32 TN`.

**Filed as:** `rocm-libraries-zvzu` — proposes extending the
quad-cycle dispatch (or `diagnose_missing_edge` directly) to
recognize LR → CVT as a TF32-emulation-expected edge.

### E. `_last_cms_capture is None` on 3 fixtures (6 entries with LDSTr variants)

**Classification: NEEDS INVESTIGATION (mechanism unclear).**

Affected: `_240x256x64_16bit NT`, `_320x192x64_16bit TN`,
`_96x256x64_16bit TN`.

CMS Build #1 did not populate `writer._last_cms_capture` despite the
runner's `try/except` not catching an exception. Possible causes:

- (H1) The kernel build raised an exception that the `except Exception:
  pass` swallowed before reaching the kernelBody post-loop assembly
  stage at `KernelWriter.py:5258+`.
- (H2) The CMS path took an early-return for these specific configs.
- (H3) Schedule registry over-reports a layout the runtime path rejects
  mid-build.

**Filed as:** `rocm-libraries-jmfp` — proposes modifying the runner
to propagate the underlying exception, then per-fixture follow-up.
Not investigated inline: each affected fixture's mechanism may be
different and requires a separate look.

### F. CaptureConsistencyError: 20-28 MFMA identities in ref but not subject (2 TF32 fixtures)

**Classification: NEEDS USER INPUT — discipline rule #8 surface.**

Affected: `_128x192x32_TF32 TN` (20 missing MFMA identities),
`_256x192x32_TF32 TN` (28 missing MFMA identities). All missing are
`v_mfma_f32_4x4x4_16b_bf16` — i.e. the **TF32 emulation
convert-pack-MFMA chain** (3 bf16 MFMAs per logical TF32 MFMA per the
"3 bf16 MFMAs per tf32 mfma" comment in
`_128x128x32_TF32.py:43`).

Three possible explanations:

- (a) Real CMS defect: schedule drops MFMAs → wrong arithmetic on
  hardware. Hard to believe never noticed, but SHADOW-vs-CMS would
  mask it (shadow shared CMS dict).
- (b) Q2-expected: per-tile schedule routes MFMAs to a different body
  / via a macro the capture pipeline doesn't observe.
- (c) Capture-pipeline bug: CMS capture doesn't observe these MFMAs
  via their actual emission path.

**Filed as:** `rocm-libraries-ldm5` (priority 0, type=question).
Recommends side-by-side asm dump on `_128x192x32_TF32 TN` (smaller
surface) to determine which explanation is correct.

---

## §4 — Sub-beads filed

| ID | Title | Class | Priority |
|---|---|---|---|
| `rocm-libraries-6hk3` | Classifier gap: VSwapB32/VPermB32/VOrB32/VMovB64 raise CaptureUnknownInstructionError | (c) | P1 |
| `rocm-libraries-zvzu` | diagnose_missing_edge: classify LR → v_cvt_pk_bf16_f32 (CVT) raw_intrawave edges | (c) | P1 |
| `rocm-libraries-p39d` | GR OrderInverted residual class generalizes beyond nyb5's 3-failure pin: extend per Q2 | (b) | P1 |
| `rocm-libraries-e293` | GRInc OrderInverted+OverriddenInputFailure pair in NGL: classify (a) real defect or (b) Q2-expected | NEEDS USER | P0 |
| `rocm-libraries-jmfp` | Investigate `_last_cms_capture is None` on 3 fixtures: real CMS build failure or early-raise | INVESTIGATE | P1 |
| `rocm-libraries-ldm5` | CaptureConsistencyError: 20-28 MFMA identities in ref but not subject for 4×4×4 mfma TF32 fixtures | NEEDS USER | P0 |

All linked to `rocm-libraries-3ija` via `discovered-from`.

---

## §5 — Implementation deltas (NONE inline)

Per the bead description: "The bead description allows implementing
**only** principled fixes that are obvious + non-controversial."

For each of the 6 residual classes:

- §3.A (GR OrderInverted in ML): no obvious one-liner. Three candidate
  approaches (extend `_NO_DATAFLOW_IDENTITY_CATEGORIES`, per-edge
  tolerance, body-blind GR identity), each with non-trivial design
  implications. Filed as `rocm-libraries-p39d`. **Did NOT remove
  the strict-XFAIL marker** in `test_approach_a_non_cms_reference.py`.
- §3.B (GRInc + SCC clobber): explicitly NEEDS USER INPUT.
- §3.C, §3.D (classifier gaps): require per-class category audit /
  dispatch-table extension. Not one-liners.
- §3.E, §3.F: explicitly NEEDS INVESTIGATION / USER INPUT.

**No production-code changes were made by 3ija.** The only artifact
landed is this memo plus the triage runner
(`Tensile/Tests/unit/_3ija_residual_triage_runner.py`). The runner is
NOT a test (it's `_`-prefixed and asserts no correctness invariants);
it's an investigation tool that the user may run manually.

---

## §6 — Open questions for the user

### Q1. Should the discipline-rule-#8 NEEDS-USER residuals (§3.B and §3.F) block 71hw landing?

Both `rocm-libraries-e293` (GRInc + SCC clobber) and
`rocm-libraries-ldm5` (missing MFMAs) potentially indicate real CMS
defects that the SHADOW path was masking. If they are real defects,
they predate any of the validator-architecture work (they shipped in
the CMS schedules already in tree) and the right next step is per-
schedule audit + fix. If they are not real defects, the validator
needs a principled extension to tolerate them.

71hw's "real-build CMS-vs-default" promise depends on Q1's answer —
because either the validator is correctly catching defects that need
fixing, or it's surfacing false positives that need a comparator
extension.

### Q2. Is the GR OrderInverted residual class (§3.A) (b)-classified?

The classification in §3.A leans (b) per nyb5's Cycle 2 mechanism
trace — but the "correct mechanism" for accepting it isn't obvious.
The 3 candidate approaches in `rocm-libraries-p39d`:

- (i) Extend `_NO_DATAFLOW_IDENTITY_CATEGORIES` with GRA/GRB. Removes
  GR identity coverage entirely. Risk: Drop too much, lose ability to
  detect e.g. CMS-emitting-fewer-GRs-than-default.
- (ii) Add per-edge tolerance: cross-build OrderInverted on GR within
  the same body is accepted. Most surgical principled fix.
- (iii) Make GR identity body-blind (loop_index ignored). Symmetric
  with hdem's MFMA body-blindness.

User decision needed before any of these lands. **Until then the
strict-XFAIL marker in `test_approach_a_non_cms_reference.py` should
remain.**

### Q3. Is the schedule registry's auto-detection probe (`dispatch.py:552`) over-reporting layouts a separate bead?

36 of 93 (fixture, layout) combos failed `_make_solution` because the
registry advertised support that the runtime Solution validator
rejects. This is not 3ija scope but it complicates any
"exhaustively exercise the CMS surface" workflow. If the user wants
the auto-detection probe to be tightened, that should be a separate
bead under the validator umbrella (or a CMS-team bead). **3ija has
NOT filed this** — flagging it here for triage.

### Q4. Are the 6 LDSTr=True fixtures with cg=0 wc=0 a regression risk?

The 6 fixtures (`_160x256x64_16bit`, `_224x256x64_16bit`,
`_240x256x64_16bit` (TN), `_256x208x64_16bit`, `_256x224x64_16bit`,
`_256x240x64_16bit`, `_256x256x64_16bit`, `_256x96x64_16bit`,
`_352x192x64_16bit`) report identical 0/0 results for both LDSTr
variants. That's a *good* result (the two builds agree) — but it
suggests LDSTr=True is producing the same captured graph as
LDSTr=False, which may itself be suspicious if the LDSTr instruction
is supposed to materially differ. Not 3ija scope to investigate, but
worth a sanity check by the CMS-schedule team.

---

## §7 — Verification before claiming done

Per the bead description's verification checklist:

- [x] Per-fixture table is complete (every gfx950 CMS schedule under
  Components/CustomSchedule/gfx950/ exercised, modulo the 36 tooling-
  artifact `_make_solution` failures and the 6 `_last_cms_capture is
  None` failures filed as `rocm-libraries-jmfp`).
- [x] Each distinct residual shape has a classification with mechanism
  citation (§3.A through §3.F).
- [x] For (a)-leaning classifications: `rocm-libraries-e293` and
  `rocm-libraries-ldm5` filed as NEEDS-USER question beads (the
  bead's "real defect" sub-class is explicitly surfaced for user
  decision rather than auto-classified).
- [x] For (b) classifications that warrant inline fix: NONE landed
  inline because no candidate fix is unambiguously
  accept-this-difference per the bead's discipline rules.
  `test_approach_a_non_cms_reference.py`'s strict-XFAIL is
  intentionally preserved.
- [x] Memo written (this file).

---

## §8 — Cross-references

- `NYB5_IMPLEMENTATION.md` — nyb5's architecture, the GRA/GRB
  Cycle 2 residual.
- `2LZD_INVESTIGATION.md §6 + §6.2` — Approach A picks + Q2 framing.
- `HDEM_IMPLEMENTATION.md` — body-blindness already provides MFMA
  cross-body tolerance; the analogous mechanism for GR is
  `rocm-libraries-p39d`'s candidate fix (iii).
- `D3ZJ_SCMPEQI32_INVESTIGATION.md` and
  `D3ZJ_NGL_NLL_LCC_INVESTIGATION.md` — d3zj's findings; nyb5
  closed the LCC residual.
- `Tensile/Components/CustomSchedule/approach_a.py:build_non_cms_reference`
  — the helper exercised by the runner.
- `Tensile/Tests/unit/_3ija_residual_triage_runner.py` — the
  investigation runner (this bead's tool).
- `Tensile/Tests/unit/test_approach_a_non_cms_reference.py:test_non_cms_reference_compare_graphs_surfaces_only_known_residuals`
  — the strict-XFAIL pin that 3ija explicitly leaves in place.
