# PGR/PLR Phase C — Design: generalize validator support across (PGR, PLR)

Bead: `rocm-libraries-ive` (sub of ot2 — Phase C). DESIGN ONLY.
This document does not modify `ScheduleCapture.py` or `KernelWriter.py`;
it specifies the change set so a follow-up implementer (or this bead's
implementation phase) can apply it mechanically.

The recommendation lives in section 1; the rationale, sketch, tests,
failure-mode analysis, and review notes follow.

---

## Status note on sibling reports (ot2.A, ot2.B)

At the time this design was drafted the two sibling reports
(`PGR_PLR_PHASE_A_REPORT.md`, `PGR_PLR_PHASE_B_REPORT.md`) had not landed
on any reachable branch — neither `users/alvasile/validator_long_term_plans`
(latest `678b75ed21`) nor any worktree-agent branch contained those
files (`git ls-tree -r users/alvasile/validator_long_term_plans
--name-only | grep PHASE_[AB]_REPORT` → empty;
`git for-each-ref refs/heads` lists no sibling branch with a Phase A/B
report subject). Sibling A/B work was therefore not directly
incorporated; all (PGR, PLR) enumeration in this doc is derived from a
direct read of `KernelWriter.py` and `ScheduleCapture.py` on
`users/alvasile/validator_long_term_plans` (commit `678b75ed21`). This
design is **preliminary in that respect — to be reconciled with sibling
reports during integration.** Specific reconciliation points are
flagged inline as **[recon-A]** / **[recon-B]**.

---

## 0. Problem statement

`KernelWriter.kernelBody` (validator-hook block,
`KernelWriter.py:5187-5266` on `users/alvasile/validator_long_term_plans`)
currently builds the default-side `FourPartCapture` like this:

```
ctx = self._capture_context
main = ctx.default_main
n_gl = ctx.default_n_gl if ctx.default_n_gl is not None else LoopBodyCapture(instructions=[])
n_ll = ctx.default_n_ll if ctx.default_n_ll is not None else LoopBodyCapture(instructions=[])
...
ctx.default = FourPartCapture(
    main_loop={0: main}, main_loop_prev={0: clone_loop_body(main)},
    n_gl={0: n_gl}, n_ll={0: n_ll},
    ...
)
```

The shadow `_noLoadLoopBodyDefault` driver in `noLoadLoop`
(`KernelWriter.py:3703-3725`) only writes to
`ctx.default_n_gl` / `ctx.default_n_ll` when it is invoked. Whether it
is invoked depends on the (PGR, PLR, SuppressNoLoadLoop, …) gating
around the production `noLoadLoop` calls (`KernelWriter.py:5118-5165`):

| Production gate | Effect on default-side capture                                  |
|-----------------|-----------------------------------------------------------------|
| `for remainPgr in range(PGR-1, 0, -1)` (5118)    | NGLL emitted iff `PGR >= 2`             |
| `if PGR and not SuppressNoLoadLoop` (5141-5142)  | NLL emitted iff `PGR >= 1` AND not suppressed |

Combined, the (PGR, SuppressNoLoadLoop) → (`default_n_gl`, `default_n_ll`)
emission matrix is:

| PGR | SuppressNoLoadLoop | `default_n_gl` populated? | `default_n_ll` populated? |
|-----|--------------------|---------------------------|---------------------------|
| 0   | any                | no                        | no                        |
| 1   | false              | no                        | yes                       |
| 1   | true               | no                        | no                        |
| ≥2  | false              | yes                       | yes                       |
| ≥2  | true               | yes                       | no                        |

PLR (`PrefetchLocalRead`, surfaced as `self.states.numItersPLR` after
config resolution) does NOT gate emission directly; it only reshapes
the contents of an emitted body. Specifically:

* CMS-side `n_gl` always expands `usePLR=1` (`expand_cms_macro` invocation
  at `ScheduleCapture.py:4710-4716`); when the kernel runs at PLR=0 the
  macro still expands but contains no LR-carry-over instructions.
* CMS-side `n_ll` always expands `usePLR=0` (`ScheduleCapture.py:4717-4723`).

Today's bug surface: when `default_n_gl` or `default_n_ll` stay None,
the consumer side replaces them with a non-None
`LoopBodyCapture(instructions=[])`. `build_dataflow_graph`
(`ScheduleCapture.py:2891-2906`) then runs:

```python
for label, by_cp in body_sources:
    if 0 not in by_cp:                  # <-- the structural absence guard
        continue
    body = by_cp[0]
    if not body.instructions:           # <-- raises CaptureEmptyBodyError
        raise CaptureEmptyBodyError(...)
```

The synthesized empty `LoopBodyCapture` defeats the `0 not in by_cp`
check (key 0 is present, mapped to an empty body) and trips the second
guard. The validator pipeline therefore raises `CaptureEmptyBodyError`
on every PGR=0 kernel and on every (PGR≥1, SuppressNoLoadLoop=true) kernel,
regardless of whether the kernel itself is correct. The validator
silently mis-fires on a structurally legitimate "no NLL/NGLL emitted"
configuration.

The consumer-facing CMS-side `FourPartCapture` does not have this
problem because `build_cms_four_part_capture`
(`ScheduleCapture.py:4727-4735`) unconditionally synthesizes both
bodies from the macro — those expansions may be substantively empty of
non-MFMA categories, but the macro always carries the MFMA backbone, so
the `body.instructions` list is never empty.

The cross-graph asymmetry — CMS always has both bodies, default has
only the bodies it actually emitted — is the root mismatch. The fix is
about making "this body wasn't emitted" representable end-to-end and
making `compare_graphs` honest about that asymmetry.

---

## 1. Recommended approach: **Hybrid (Option 3)**

### 1.1 Capture-side honesty (Option 1 component)

In `KernelWriter.kernelBody`, replace the `if … is not None else
LoopBodyCapture(instructions=[])` synthesis with the structural-absence
encoding `build_dataflow_graph` already understands: omit the key
entirely when no body was emitted.

```
n_gl_dict = {0: ctx.default_n_gl} if ctx.default_n_gl is not None else {}
n_ll_dict = {0: ctx.default_n_ll} if ctx.default_n_ll is not None else {}
ctx.default = FourPartCapture(
    main_loop={0: main}, main_loop_prev={0: clone_loop_body(main)},
    n_gl=n_gl_dict, n_ll=n_ll_dict,
    ...
)
```

This is a one-site edit; the rest of `FourPartCapture` already accepts
arbitrary dicts (the dataclass typing is `dict`, not `Dict[int, ...]`).

### 1.2 CMS-side conditional construction (mirror)

`build_cms_four_part_capture` (`ScheduleCapture.py:4679-4736`) must
match: skip the macro expansion that has no default-side counterpart,
so the CMS graph doesn't synthesize a phantom NGL/NLL body the default
side can't possibly equal. The trigger is *not* PGR/PLR (the macro
itself doesn't know — it's expanded post-codegen); rather, the call
site must pass an explicit `emit_n_gl: bool` / `emit_n_ll: bool`. The
two flags come from the same kernel-config predicates that gate the
default-side `noLoadLoop` calls (see 1.3).

```
def build_cms_four_part_capture(macro, num_codepaths, ...,
                                emit_n_gl=True, emit_n_ll=True):
    ...
    n_gl = {0: expand_cms_macro(...)} if emit_n_gl else {}
    n_ll = {0: expand_cms_macro(...)} if emit_n_ll else {}
    return FourPartCapture(..., n_gl=n_gl, n_ll=n_ll, ...)
```

Default values preserve existing test fixtures (every direct caller in
`test_ScheduleCapture.py` keeps both bodies populated by passing
nothing).

### 1.3 Structural predicate (kernel-config-derived)

The single predicate that controls both default-side emission and the
new CMS-side flags is read off `kernel`:

```
def kernel_emits_n_gl(kernel) -> bool:
    return kernel["PrefetchGlobalRead"] >= 2

def kernel_emits_n_ll(kernel) -> bool:
    return bool(kernel["PrefetchGlobalRead"]) and not kernel["SuppressNoLoadLoop"]
```

These are the *exact* boolean reductions of the `for remainPgr in
range(PGR-1, 0, -1)` and `if PGR and not SuppressNoLoadLoop:` gates at
`KernelWriter.py:5118` and `KernelWriter.py:5141-5142`. They live in
`ScheduleCapture.py` (top-level helpers, near
`build_cms_four_part_capture`) so the CMS and default sides import the
same predicate — a single source of truth that stays grep-able if the
production gates ever shift. The default-side `kernelBody` block can
*also* call them, asserting the captured state matches the predicted
state (see 1.5). The validator hook then becomes:

```
emit_n_gl = kernel_emits_n_gl(kernel)
emit_n_ll = kernel_emits_n_ll(kernel)
n_gl_dict = {0: ctx.default_n_gl} if emit_n_gl else {}
n_ll_dict = {0: ctx.default_n_ll} if emit_n_ll else {}
assert (ctx.default_n_gl is not None) == emit_n_gl, (...)
assert (ctx.default_n_ll is not None) == emit_n_ll, (...)
```

### 1.4 Graph-build-side guard relax (Option 2 component)

`build_dataflow_graph` (`ScheduleCapture.py:2848-2906`) is left
substantively unchanged — `if 0 not in by_cp: continue` already does
the structurally-correct thing once 1.1+1.2 are in place. The only
edit is to its docstring (lines 2879-2884): the line "Always raises
CaptureEmptyBodyError if any body has zero instructions" stays, but a
new line is added: "An entirely-omitted body (key 0 absent) is treated
as 'this body was not emitted by either scheduler', NOT as an error."
This is a doc clarification, not a behavior change — but it pins the
contract so the next time someone tries to "fix" the structural-absence
path by synthesizing an empty body, the docstring talks them out of it.

### 1.5 Sanity-check at graph-build time (Option 3 component)

A new small helper in `ScheduleCapture.py`:

```
def assert_capture_body_consistency(four_part_capture, kernel):
    """Cross-check the FourPartCapture's body presence against the
    kernel-config-derived predicates. Raises CaptureConsistencyError on
    mismatch — would catch e.g. a future production-side gate change
    that quietly stops emitting NGLL while the predicate still says it
    should."""
    expected_n_gl = kernel_emits_n_gl(kernel)
    expected_n_ll = kernel_emits_n_ll(kernel)
    actual_n_gl  = 0 in four_part_capture.n_gl
    actual_n_ll  = 0 in four_part_capture.n_ll
    if actual_n_gl != expected_n_gl or actual_n_ll != expected_n_ll:
        raise CaptureConsistencyError(...)
```

`kernelBody` calls this on `ctx.default` AND `ctx.cms` immediately
after both are built, before `build_dataflow_graph`. CaptureConsistencyError
already exists (`ScheduleCapture.py:79`) — its docstring widens to
include this case.

`compare_graphs` itself does **not** need a new "asymmetric body
presence" code path because both ref and subj graphs will agree on
which body labels exist (both honor the same predicate). If a future
change makes them diverge, the new sanity-check fires before
`compare_graphs` runs.

---

## 2. Rationale

### 2.1 Why hybrid, not pure capture-side

Option 1 alone (only fix the consumer-side dict construction) leaves a
silent class of latent bugs: if a future PGR/PLR/SNLL gate change
makes the default-side stop emitting a body that *the kernel-config
predicate says should exist*, the validator silently passes (because
both sides agree the body is absent). The Option 3 sanity-check turns
that silent skip into a loud `CaptureConsistencyError`.

### 2.2 Why not pure graph-build-side

Option 2 alone (relax `build_dataflow_graph` to treat
`{0: empty_body}` as "no body emitted") is cheap but conflates two
distinct error modes: "the body was never emitted" (legitimate under
PGR=0) and "the body was emitted but the capture pipeline lost all its
instructions" (a real bug — the latter is exactly what
`CaptureEmptyBodyError` was added to surface). Collapsing them would
hide a regression class. The hybrid keeps both errors distinct: empty
list = lost data (raises), absent key = not emitted (skips), and the
sanity-check at 1.5 prevents either side from synthesizing a
mismatched key.

### 2.3 Why the predicate lives in ScheduleCapture, not KernelWriter

The predicate is the **contract** the validator pipeline depends on.
Putting it next to `build_cms_four_part_capture` means the CMS-side
constructor and the production-side code path read the same lines.
KernelWriter still owns the gating logic; ScheduleCapture exposes the
boolean reduction the validator needs.

### 2.4 Why not introduce a "body=None" sentinel instead of dict-omission

The `FourPartCapture` dict-of-bodies shape was already chosen
(`ScheduleCapture.py:287-322`) to permit per-codepath omission for
main_loop. Reusing the same "key absent" idiom for n_gl/n_ll keeps the
data model uniform and avoids adding an `Optional[LoopBodyCapture]`
wrapper for one specific code path. `build_dataflow_graph` already
handles the absent-key case correctly (line 2898).

---

## 3. Implementation sketch (file:line targets)

All references on `users/alvasile/validator_long_term_plans` @ `678b75ed21`.

### 3.1 `Tensile/Components/ScheduleCapture.py`

* **Insert** (top-level, immediately before `build_cms_four_part_capture`,
  ~line 4677): two new public predicate helpers.

  ```
  def kernel_emits_n_gl(kernel) -> bool:
      return kernel["PrefetchGlobalRead"] >= 2
  def kernel_emits_n_ll(kernel) -> bool:
      return bool(kernel["PrefetchGlobalRead"]) and not kernel["SuppressNoLoadLoop"]
  ```

  Each helper carries a docstring naming the production gate it
  mirrors (`KernelWriter.py:5118` / `KernelWriter.py:5141-5142`) so
  cross-references stay grep-able.

* **Insert** (top-level, near the predicates): the new sanity helper.

  ```
  def assert_capture_body_consistency(four_part_capture, kernel) -> None:
      ...  # raises CaptureConsistencyError on mismatch
  ```

* **Edit** `build_cms_four_part_capture`
  (`ScheduleCapture.py:4679-4736`): add `emit_n_gl=True, emit_n_ll=True`
  kwargs; gate `n_gl_body` / `n_ll_body` expansion on those flags;
  return `n_gl={0: n_gl_body} if emit_n_gl else {}` (mirror for n_ll).

* **Edit** `CaptureConsistencyError` docstring
  (`ScheduleCapture.py:79-80`): widen to mention body-presence
  consistency between predicate and capture.

* **Edit** `build_dataflow_graph` docstring (`ScheduleCapture.py:2879-2884`):
  add the clarification spelled out in 1.4. No code change.

### 3.2 `Tensile/KernelWriter.py`

* **Edit** the validator-hook block at `KernelWriter.py:5187-5266`,
  specifically lines 5206-5207 (current `if … is not None else
  LoopBodyCapture(instructions=[])` synthesis) and the `FourPartCapture`
  construction at 5217-5227. New shape:

  ```
  emit_n_gl = kernel_emits_n_gl(kernel)
  emit_n_ll = kernel_emits_n_ll(kernel)
  assert (ctx.default_n_gl is not None) == emit_n_gl, (...)
  assert (ctx.default_n_ll is not None) == emit_n_ll, (...)
  n_gl_dict = {0: ctx.default_n_gl} if emit_n_gl else {}
  n_ll_dict = {0: ctx.default_n_ll} if emit_n_ll else {}
  ctx.default = FourPartCapture(
      main_loop={0: main},
      main_loop_prev={0: clone_loop_body(main)},
      n_gl=n_gl_dict, n_ll=n_ll_dict,
      ...
  )
  assert_capture_body_consistency(ctx.default, kernel)
  if ctx.cms is not None:
      assert_capture_body_consistency(ctx.cms, kernel)
  ```

* The CMS-side construction site (the call to
  `build_cms_four_part_capture` from `customMainLoopSchedule` /
  `simdSpecDispatch` machinery — outside the worktree-visible call sites,
  resolved via grep on `users/alvasile/validator_long_term_plans`) needs
  to pass `emit_n_gl=kernel_emits_n_gl(kernel)`,
  `emit_n_ll=kernel_emits_n_ll(kernel)`. **[recon-A]**: sibling A's
  enumeration should pin the exact call site; if A reports more than
  one CMS-side construction call, all of them need the same kwargs.

### 3.3 `Tensile/Tests/unit/test_ScheduleCapture.py`

See section 4 — new test class `TestPgrPlrCaptureMatrix` covering the
matrix rows.

---

## 4. New test cases (one per matrix row)

Test class: `TestPgrPlrCaptureMatrix` in
`Tensile/Tests/unit/test_ScheduleCapture.py`.

Each row asserts (a) presence/absence of n_gl/n_ll keys in
`ctx.default` and `ctx.cms`, (b) `build_dataflow_graph` succeeds (no
`CaptureEmptyBodyError`), (c) `assert_capture_body_consistency` does
not raise, (d) `compare_graphs` returns zero failures on a clean
kernel.

| # | PGR | SuppressNoLoadLoop | PLR | Expected `default.n_gl`? | Expected `default.n_ll`? | Test name (proposed) |
|---|-----|--------------------|-----|--------------------------|--------------------------|----------------------|
| 1 | 0   | false              | 0   | absent                   | absent                   | `test_matrix_pgr0_plr0` |
| 2 | 0   | false              | 1   | absent                   | absent                   | `test_matrix_pgr0_plr1` |
| 3 | 1   | false              | 0   | absent                   | present                  | `test_matrix_pgr1_plr0` |
| 4 | 1   | false              | 1   | absent                   | present                  | `test_matrix_pgr1_plr1` |
| 5 | 1   | true               | 1   | absent                   | absent                   | `test_matrix_pgr1_snll_plr1` |
| 6 | 2   | false              | 0   | present                  | present                  | `test_matrix_pgr2_plr0` |
| 7 | 2   | false              | 1   | present                  | present                  | `test_matrix_pgr2_plr1` |
| 8 | 2   | true               | 1   | present                  | absent                   | `test_matrix_pgr2_snll_plr1` |
| 9 | 3   | false              | 1   | present                  | present                  | `test_matrix_pgr3_plr1` |

Two unit-level invariants worth their own tests:

* `test_predicate_matches_production_gate`: parametrize over the matrix
  rows; assert `kernel_emits_n_gl(kernel)` matches whether
  `ctx.default_n_gl` is set after a real build (the truth-table the
  predicate is supposed to encode).
* `test_consistency_check_catches_predicate_drift`: monkeypatch the
  predicate to flip a bit; assert `assert_capture_body_consistency`
  raises `CaptureConsistencyError`. Pins the sanity-check.

PLR=0/PLR=1 coverage in rows 1/2 and 3/4 and 6/7 is intentional even
though PLR doesn't gate emission — it gates body **contents**, so a
PLR=0 case where n_gl is present must still end up with a non-empty
body (because the macro carries MFMAs even with usePLR=0). If a PLR=0
build ever produces an empty `n_gl` body, the existing
`CaptureEmptyBodyError` (which we deliberately keep) fires — that's
the catch path.

PGR=3 (row 9) is the high-PGR loop-iteration case
(`for remainPgr in range(PGR-1, 0, -1)` runs twice); the predicate
collapses it to "n_gl present" but the production loop emits NGLL
*twice*. Capture currently overwrites — the second invocation wins
(`ctx.default_n_gl = finalized` at `KernelWriter.py:3723`). This is
out-of-scope for Phase C (it's a separate "multi-NGLL collapse"
question), but row 9 pins the behavior so a future change cannot
silently start aggregating without a test failure. **[recon-B]**:
sibling B's coverage probe likely already enumerated which gfx1151
yamls hit PGR≥3; its findings should be cross-referenced for the
expected populated-body length.

### 4.1 Migration of existing tests

* `test_n_gl_and_n_ll_populated` (`test_ScheduleCapture.py:1056-1073`)
  uses a kernel built with `PrefetchGlobalRead: 2` (line 1125) — both
  n_gl and n_ll should be populated under the new design. **No
  migration needed** for this test (it exercises matrix row 7).

* `test_n_gl_n_ll_state_resets_after_kernel`
  (`test_ScheduleCapture.py:1086-1103`) asserts that
  `ctx.default_n_gl` is None after `reset()`. This is about the
  scratch-state reset semantics, not the FourPartCapture construction.
  **No migration needed** — `CaptureContext.reset()` (lines 273-281)
  still nulls the slots; only the way `kernelBody` reads them changes.

* `TestDataflowGraphIntegration.test_dataflow_gating_passes_on_clean_cms_kernel`
  (line 1144) and the `MIArchVgpr=True` variant (1154) both build with
  PGR=2 — already exercise rows 7 and (with override) 7 again. **No
  migration needed**, but a parametric variant covering rows 1/3/5 (the
  empty-body cases the current code mis-handles) is the load-bearing
  new coverage.

* `test_FourPartCapture_default_construction` etc.
  (`test_ScheduleCapture.py:198-228`) construct FourPartCapture
  directly with all four bodies populated. With the dict-omission
  encoding, **add** parametric variants where `n_gl={}` and `n_ll={}`
  to pin that the dataclass and `build_dataflow_graph` accept that
  shape end-to-end. Existing tests stay unchanged.

* `build_cms_four_part_capture` callers in tests (none currently
  exercise the new kwargs) — default kwargs preserve historical
  behavior. **No migration needed.**

---

## 5. Failure-mode analysis

### 5.1 Bugs the design CATCHES that current behavior misses

* **The original Phase C bug**: PGR=0 / (PGR=1 ∧ SNLL) / (PGR≥2 ∧ SNLL)
  kernels currently hit `CaptureEmptyBodyError` on every build (false
  positive). The dict-omission encoding makes the validator silent on
  legitimate non-emission. Net: enables validator coverage for a
  whole class of kernel configurations that today must disable
  `_captureDefaultSchedule` to build at all.

* **Predicate drift**: if the production gates at `KernelWriter.py:5118`
  / `5141-5142` change (e.g. someone adds a new
  `if not kernel["DTLA"]:` short-circuit) without updating
  `kernel_emits_n_gl/n_ll`, the new
  `assert_capture_body_consistency` check fires loudly with both the
  predicted and observed body presence. Today the equivalent
  divergence would either silently skip validation (n_gl = empty,
  comparison passes vacuously) or raise an unhelpful
  `CaptureEmptyBodyError`.

* **CMS-side phantom body**: if `build_cms_four_part_capture` is ever
  called for a PGR=0 kernel (today: would happen iff someone forgets
  the new `emit_n_gl=False` kwarg), the consistency check on the CMS
  graph fires before `compare_graphs` does. Today: would compare a
  CMS-synthesized n_gl/n_ll against an absent default-side body and
  produce a long list of spurious "missing edge" failures.

### 5.2 Bugs the design RISKS letting through compared to today

* **Genuine empty-body bug masked as "not emitted"**: if the production
  default-side path is supposed to emit n_gl (predicate says yes) but
  actually emits zero instructions due to a real bug in
  `_noLoadLoopBodyDefault`, today's behavior raises
  `CaptureEmptyBodyError`. The new design also raises (we KEEP the
  empty-list guard at `ScheduleCapture.py:2901-2905` — the only thing
  that changes is the dict-omission path). **Risk: NONE — explicitly
  preserved by the hybrid design.**

* **Predicate vs production gate skew**: if the kernel-config predicate
  is wrong (says yes when production says no, or vice versa), the
  consistency check fires. **Risk: a wrong predicate becomes a hard
  failure rather than silent over/undervalidation.** This is desirable
  but means any production gate refactor must be paired with a
  predicate update — call this out in the PR description.

* **Multi-NGLL aggregation for PGR≥3**: the current
  `default_n_gl = finalized` overwrite (`KernelWriter.py:3723`) loses
  all but the last NGLL invocation. Phase C explicitly does not fix
  this; the predicate collapses PGR=2 and PGR=3 to the same "n_gl
  present" answer. **Risk: a real PGR=3 NGLL-pair divergence would not
  surface.** Mitigation: row 9 of the test matrix is a pin (asserts the
  capture is non-empty for PGR=3); follow-up bead can extend
  `default_n_gl` to a list-of-bodies-by-NGLLindex.

* **Tests that built directly against the synthesized empty-body shape**:
  none found in `test_ScheduleCapture.py` (verified via grep for
  `LoopBodyCapture(instructions=[])` — only the consumer site at
  `KernelWriter.py:5206-5207` constructs that shape). **Risk:
  effectively zero.**

### 5.3 Edge cases walked through

* **`SuppressNoLoadLoop=True` ∧ PGR=0**: `n_ll` predicate returns false
  (PGR=0 short-circuits), `n_gl` predicate returns false (PGR<2). Both
  bodies absent — consistent with the production behavior (no NLL/NGLL
  emitted). No capture, no validation gap (compare_graphs walks an
  empty body label set for that side; today's CMS side would also be
  empty under the new emit_n_gl/emit_n_ll plumbing).

* **`useTailloopInNll=True`** (`KernelWriter.py:5167-5185`): emits a
  *second* NLL for the tailloop-not-applicable branch. Both NLL
  emissions overwrite `default_n_ll` — same multi-overwrite issue as
  PGR≥3 for n_gl. Predicate still says "n_ll present". Existing tests
  do not exercise tailloopInNll under capture; the new test matrix
  does not either. **Misfit flag: tailloopInNll is silently
  collapsed.** Recommendation: out-of-scope follow-up bead, mirror of
  the PGR≥3 follow-up.

* **`isOptNLL=True`** (`KernelWriter.py:3703`): the shadow capture is
  guarded `not isOptNLL`. OptNLL paths would not populate
  `default_n_ll`. Predicate currently doesn't model OptNLL; would
  produce a false consistency-check failure for kernels that are
  PGR≥1 ∧ OptNLL. **Misfit flag: predicate must include `not
  kernel["OptNoLoadLoop"]` (or equivalent) — sibling A's enumeration
  should pin the exact kernel-key name. [recon-A]** Tentatively the
  predicate becomes:

  ```
  def kernel_emits_n_ll(kernel) -> bool:
      if kernel.get("OptNoLoadLoop"):  # OptNLL path is its own emission
          return False
      return bool(kernel["PrefetchGlobalRead"]) and not kernel["SuppressNoLoadLoop"]
  ```

  Until [recon-A] resolves, the Phase C implementer should land the
  PGR/SuppressNoLoadLoop predicate first and iterate on OptNLL handling
  in a follow-up. The consistency check makes this safe — it'll raise
  loudly on the first OptNLL kernel that hits the validator.

* **DTV / DirectToVgpr odd/even NLL**
  (`KernelWriter.py:5147-5165`, `NeedNLLOddEven = isDTV`): emits
  TWO NLLs (odd and even). Same overwrite issue. **Misfit flag.**
  Same out-of-scope follow-up.

* **`needSecondNGLL`** (line 5121-5125): emits a second NGLL for the
  second GR buffer (`UnrollLoopSwapGlobalReadOrder` path). Same
  overwrite issue. **Misfit flag.**

* **Multi-loopCopies (loopCopies=2)**: gated out by the existing
  `assert loopCopies == 1` (`KernelWriter.py:5198-5202`). Phase C does
  not change this; the predicate ↔ capture invariant only holds for
  single-loop-copy CMS kernels.

The four "Misfit flag" cases (tailloopInNll, OptNLL, DTV
odd/even, needSecondNGLL) all share the same shape: production
emits *multiple* NGLL/NLL bodies, capture overwrites and keeps only the
last one. Phase C's predicate is a 1-bit "present or not" answer;
representing 2-or-more-body shapes requires extending
`default_n_gl`/`default_n_ll` from a single `LoopBodyCapture` slot to a
list. Explicitly out of scope; the test matrix pins the 1-bit answer
so the follow-up can extend without breaking existing tests.

---

## 6. Sources

* `KernelWriter.py:5118-5165` (NGLL/NLL emission gates) — direct read
  on `users/alvasile/validator_long_term_plans` @ `678b75ed21`.
* `KernelWriter.py:5187-5266` (validator-hook block, current empty-body
  synthesis) — direct read.
* `KernelWriter.py:3703-3725` (shadow-capture driver in `noLoadLoop`)
  — direct read.
* `ScheduleCapture.py:2848-2906` (`build_dataflow_graph`, structural
  absence vs empty-body distinction) — direct read.
* `ScheduleCapture.py:4679-4736` (`build_cms_four_part_capture`) —
  direct read.
* `ScheduleCapture.py:79-93` (`CaptureEmptyBodyError`,
  `CaptureConsistencyError`) — direct read.
* `Tests/unit/test_ScheduleCapture.py:1056-1103` (existing Phase 5
  capture tests) — direct read.
* Sibling reports (`PGR_PLR_PHASE_A_REPORT.md`,
  `PGR_PLR_PHASE_B_REPORT.md`): **not reachable** at draft time;
  reconciliation flagged at **[recon-A]** (kernel-config key names for
  OptNLL / DTV / SuppressNoLoadLoop, exact CMS-side
  `build_cms_four_part_capture` call site count) and **[recon-B]**
  (which yaml/MT shapes in the gfx1151 audit corpus exercise PGR=0,
  PGR=1, PGR≥3 — ground for Section 4 row coverage).

---

## 7. Critical-review pass

A general-purpose reviewer subagent was *not* spawnable from this
context (the harness exposes Skill / Bash / Read / Edit / Write /
Monitor / WebFetch / WebSearch / Notebook tools, not a `subagent_type`
dispatch). In lieu of a spawned reviewer, the author conducted a
self-critical adversarial pass with explicit "what would a reviewer
flag?" framing. Findings and resolutions:

1. **"You assumed PLR doesn't gate emission — verify."**
   Re-read `KernelWriter.py:5118-5165`. Confirmed: `PrefetchLocalRead`
   appears nowhere in the surrounding `if/for` headers; only PGR and
   SuppressNoLoadLoop. `numItersPLR` (PLR derived) is read inside
   `_noLoadLoopBodyDefault` for body shape (line 3109-3110), not
   gating. Held.

2. **"`expand_cms_macro` may emit zero instructions for some
   (usePLR, useGR) combo, defeating the empty-body guard."**
   Re-read `expand_cms_macro` indirectly via `build_cms_four_part_capture`:
   the macro always carries the MFMA backbone (line 4725 counts MFMAs
   on the main_loop body, requiring it to be non-empty as a downstream
   invariant). For n_gl: usePLR=1 keeps LR, usePLR=0 strips them; MFMA
   is unaffected by both flags. So the body always has at least the
   MFMA stream. Confirmed; the empty-body guard remains the right
   semantic for "this body should have content but doesn't".

3. **"What if the predicate is computed from `kernel` BEFORE the
   solution-resolution pass that may rewrite SuppressNoLoadLoop?"**
   The validator hook at `KernelWriter.py:5192` runs from `kernelBody`,
   which is downstream of `setupNewTile`. By that point `kernel` is
   the resolved Solution dict. Verified by inspection:
   `_captureDefaultSchedule` itself is set inside `kernelBody` at line
   4692-4693 reading `kernel.get("UseCustomMainLoopSchedule")`. Same
   read pattern. Held.

4. **"Multiple-NGLL-bodies (PGR≥3) overwrite — your predicate gives a
   single-bit answer; that's wrong."**
   Acknowledged and explicitly flagged in 5.3. The single-bit answer
   is correct *for the consistency check today*; representing
   multi-body capture requires data-model extension (slot → list)
   which is out-of-Phase-C scope. The test matrix row 9 pins the
   1-bit answer so a future fix doesn't silently regress.

5. **"The CMS-side construction call sites — you waved at 'outside the
   worktree-visible call sites'. Where exactly?"**
   `build_cms_four_part_capture` is grep-target. On
   `users/alvasile/validator_long_term_plans` the only test-vs-prod
   distinction is: tests construct `FourPartCapture` directly;
   production goes through `build_cms_four_part_capture` from a single
   site I have not yet pinpointed (the search would require expanding
   sparse-checkout or scanning the full tree). **Action:** the Phase C
   implementer must run `git grep build_cms_four_part_capture` on the
   full tree to enumerate call sites; if more than one, all need the
   new kwargs. Already noted as **[recon-A]** in 3.2.

6. **"Your `kernel_emits_n_ll` ignores OptNLL — false positive on every
   OptNLL kernel."**
   Acknowledged as misfit in 5.3 with the proposed predicate
   refinement. Phase C implementer should ship the simple form first;
   the consistency check makes the OptNLL gap loud rather than silent.

7. **"What if a kernel has PGR=1 and SuppressNoLoadLoop=True — your
   table says no n_gl AND no n_ll. Does the validator have anything to
   compare?"**
   Yes: main_loop is always populated. With both n_gl and n_ll absent,
   `build_dataflow_graph` walks main_loop only. `compare_graphs`
   compares main_loop graphs across ref/subj. This is degraded
   coverage (no tail-loop validation) but better than today's hard
   failure. The audit doc should record that PGR≤1∧SNLL kernels get
   "main-loop-only" validation.

8. **"`assert_capture_body_consistency` runs on `ctx.cms` — but what
   if CMS capture wasn't created (non-CMS path)?"**
   `kernelBody`'s validator-hook block at lines 5230-5232 already
   checks `if ctx.cms is not None:` before running compare_graphs. The
   new consistency-check call is gated the same way (see the snippet
   in 3.2: `if ctx.cms is not None: assert_capture_body_consistency(...)`).
   Held.

9. **"Your test row 5 (PGR=1, SNLL=True) — does
   `_build_with_capture` actually accept `SuppressNoLoadLoop` as an
   override?"**
   `_build_with_capture` calls `_make_solution(config, …)` with
   arbitrary overrides; SuppressNoLoadLoop is a standard Solution
   parameter (used at `KernelWriter.py:5142`). Should pass through
   unchanged. **Verification step for the implementer**: run
   `test_matrix_pgr1_snll_plr1` first; if Solution rejects
   SuppressNoLoadLoop=True with PGR=1 (some Solution-validation rules
   couple them), use the next-cleanest matrix point.

10. **"You claim no test currently constructs the synthesized
    `LoopBodyCapture(instructions=[])` shape — did you check beyond
    `test_ScheduleCapture.py`?"**
    `git grep "LoopBodyCapture(instructions=\[\])"` on
    `users/alvasile/validator_long_term_plans` returns only
    `KernelWriter.py:5206-5207` (the two consumer-side lines this
    design replaces). Verified.

Self-review changes incorporated above:

* Added the OptNLL refinement and explicit out-of-scope follow-up
  list in 5.3.
* Pinned the consistency-check gating on `ctx.cms` in 3.2 and 7-item-8.
* Added the verification step for SuppressNoLoadLoop+PGR=1 to row 5
  (item 9 above).
* Added the "main-loop-only validation" coverage downgrade note for
  PGR≤1∧SNLL kernels (item 7 above) — record this in the audit doc.
