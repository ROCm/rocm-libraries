# nyb5: Approach-A non-CMS reference build (foundational helper)

This memo documents `rocm-libraries-nyb5`, the foundational sub-bead of
`rocm-libraries-71hw` (the meta-investigation that delivers Approach A
real-build CMS-vs-default validation per `2LZD_INVESTIGATION.md §6`).

## What landed

### Helper signature + location

```python
# Tensile/Components/CustomSchedule/approach_a.py
def build_non_cms_reference(kernel_config, asm, isaInfoMap) -> FourPartCapture:
    ...
```

The helper:

1. Deep-copies `kernel_config` so the caller's dict is unaffected.
2. Forces `UseCustomMainLoopSchedule=0` on the copy.
3. Builds a `Solution` via `cms_test_utils._make_solution` (the same
   helper the existing fixtures use — single source of truth for solution
   construction in tests).
4. Spins up a fresh `KernelWriterAssembly` instance (Q5 — second writer
   instance, fully isolated; mirrors `_dump_carveout_assembly.py:229`).
5. Calls `writer.enable_capture_non_cms_build()` to switch on the new
   capture path (gated separately from the legacy SHADOW
   `_captureDefaultSchedule` flag).
6. Calls `writer._getKernelSource(solution)` — the standard production
   entry point. The non-CMS branches of `_loopBody` and `noLoadLoop`
   feed `_capture_context.builder` / `default_n_gl` / `default_n_ll`.
7. Returns `writer._last_default_capture` (a `FourPartCapture` with
   `source="non-cms-reference"`, distinguishing it from the legacy
   shadow capture's `"default-sia3"`).

### Integration sites (KernelWriter.py)

All sites are ADDITIVE — every existing `_captureDefaultSchedule`
flag-check site (9 sites verified: lines 3858, 4540, 4634, 4728, 4755,
5142, 5242, 5300, 5458) and the auto-activation at line 4918 remain
untouched. The shadow-capture extraction code paths
(`_capture_context.default`/`cms` lifecycle, `compare_graphs` /
`validate_edge_wait_coverage` invocation) are unchanged.

The new `_captureNonCmsBuild` flag drives:

- **Public lifecycle hook** (KernelWriter.py:526-572): new method
  `enable_capture_non_cms_build()`. Mirrors the
  `enable_capture_default_schedule` pattern — monkey-patches
  `setupNewTile` once with a wrapper that sets the state flag. Idempotent.
- **`_loopBody` non-CMS branch** (KernelWriter.py:4568-4632): when the
  flag is set, lazy-create `_capture_context.builder`, build a
  `capture_id_to_category` map via `build_idmap` /
  `invert_idmap_to_id_to_category` (the single source of truth for
  category tagging), and pass `capture=builder, capture_id_to_category=...`
  to `_makeSubIterSchedule`. Same call shape as the SHADOW path's elif
  branch above it — just on a different flag and consuming the actual
  natural-emit modules instead of `structural_clone`s.
- **`_loopBody` post-closeLoop hook** (KernelWriter.py:4794-4811): on
  the non-CMS path, after `closeLoop` returns its module, invoke
  `_appendCloseLoopLCCToBuilder` to harvest the loop-counter code (LCC
  — `SSubU32` + `SCmpEQI32` targeting `LoopCounterL`), then finalize
  `_capture_context.builder` into `_capture_context.default_main`.
  This is the structural fix for d3zj — see "Verification" below.
- **`_appendCloseLoopLCCToBuilder` helper** (KernelWriter.py:2732-2789):
  walks `closeLoopModule.flatitems()`, appends every `SSubU32` /
  `SCmpEQI32` to the builder under `category="LCC"`. The helper does
  NOT distinguish LoopCounterL from other SSubU32 destinations — it
  trusts that closeLoop only emits LCC-side `SSubU32` (verified against
  `KernelWriterAssembly.py:6850-6892`; ShadowLimit decrements happen
  elsewhere in `globalReadIncrement*`).
- **`noLoadLoop` non-CMS capture** (KernelWriter.py:3892-3914): when
  the flag is set AND `not isOptNLL` AND we're on the non-CMS path,
  build a fresh `LoopBodyCaptureBuilder`, call `_noLoadLoopBodyDefault`
  with `capture=builder`, finalize, stash on
  `_capture_context.default_n_gl` / `default_n_ll`. Mirrors the SHADOW
  path's noLoadLoop driver (lines ~3858-3880) but consumes the
  natural emit (no clone needed).
- **`kernelBody` post-loop assembly** (KernelWriter.py:5616-5666): a
  parallel branch gated on `_captureNonCmsBuild` that assembles
  `ctx.default = FourPartCapture(...)` with `source="non-cms-reference"`.
  Does NOT run `compare_graphs` or `validate_edge_wait_coverage` — the
  whole point of Approach A is that the comparison happens OUTSIDE the
  build, by the validator orchestrating Build #1 (CMS) and Build #2
  (this) captures. The same RegSet-stream harvest +
  `name_to_idx` seeding pattern as the legacy site keeps symbolic
  operands resolving consistently across both builds (the load-bearing
  invariant for `compare_graphs` per `rocm-libraries-bb34`).

### Test surface

`Tests/unit/test_approach_a_non_cms_reference.py` (4 tests):

1. `test_build_non_cms_reference_returns_body_shaped_capture` (Cycle 1
   GREEN): asserts the helper returns a capture with ML-1, ML, NGL, NLL
   each populated and each with a non-zero data-flow instruction count.
2. `test_non_cms_reference_compares_clean_against_cms_build` (Cycle 2
   STRICT-XFAIL): the `compare_graphs(ref=Build#2, subj=Build#1)` call
   surfaces 3 `OrderInvertedFailure`s on GRA/GRB nodes in the ML body.
   See "Verification — Cycle 2 surprise" below.
3. `test_non_cms_reference_compare_graphs_surfaces_only_known_residuals`
   (PASSED): pins the EXACT residual surfaced by Cycle 2 (3
   OrderInvertedFailures, all GRA/GRB, all in body=ML) so any new kind
   of failure escalates as a test failure rather than a silent shape
   change.
4. `test_non_cms_reference_has_lcc_in_every_main_loop_body` (Cycle 3
   GREEN): asserts ML-1 and ML each have exactly 1 SSubU32 + 1
   SCmpEQI32 tagged `category="LCC"`. This is the d3zj defect closing
   at the helper level.

Test count delta: +4 tests (3 PASSED, 1 strict-XFAIL).

## Isolation strategy (Q5)

Per `2LZD_INVESTIGATION.md §6.2 Q5`, the implementation uses a SECOND
writer instance for Build #2, fully isolated from Build #1. The pattern
mirrors `Tests/unit/_dump_carveout_assembly.py:226-230`:

```python
# Build #2: default path. UseCustomMainLoopSchedule=0 bypasses the
# CMS dispatch entirely (no per-tile schedule registration fires, no
# UsePLRPack/UseMFMAF32XEmulation mutation), so we get a true non-CMS
# Tensile emit for the same problem shape.
default_config = dict(CANONICAL_KERNEL_CONFIG)
default_config['UseCustomMainLoopSchedule'] = 0
default_solution = _make_solution(default_config, asm, isaInfoMap)
default_writer = KernelWriterAssembly(asm, DebugConfig())
default_asm_text = default_writer._getKernelSource(default_solution)
```

`build_non_cms_reference` follows the same pattern: separate config copy,
separate solution, separate writer. Zero state contamination between the
two builds.

### Shared resource audit

The two writers share:
- `Assembler` instance (`asm`) — read-only state; both writers consume
  it but neither mutates it.
- `isaInfoMap` — read-only.
- The process-global `rocIsa` singleton — set by the assembler init;
  unchanged across builds.

No mutable shared state surfaced during implementation. If a future
audit identifies a shared resource that genuinely cannot be isolated
(per the bead's "STOP and surface" rule on Q5), it should be re-raised
under `rocm-libraries-71hw`.

## Verification

### Cycle outcomes (TDD discipline)

- **Cycle 1 RED** captured verbatim:
  ```
  E   ModuleNotFoundError: No module named 'Tensile.Components.CustomSchedule.approach_a'
  ```
  GREEN: helper module created, returns `FourPartCapture` with all four
  bodies populated.
- **Cycle 2 RED** captured verbatim:
  ```
  E   AssertionError: Approach-A compare_graphs surfaced 3 failure(s)
      with the non-CMS reference; first 3:
      ['OrderInvertedFailure', 'OrderInvertedFailure', 'OrderInvertedFailure']
  ```
  GREEN: see "Cycle 2 surprise" below — the bead's STOP-and-surface
  rule applies. The Cycle 2 test is now strict-XFAIL pinning the
  residual; a companion test pins that the residual is exactly the
  GRA/GRB-in-ML shape (no other kinds).
- **Cycle 3 RED** captured verbatim:
  ```
  E   AssertionError: Per-body LCC invariant violation in non-CMS reference:
      ['ML-1: SSubU32=3 SCmpEQI32=1 (expected 1 of each)',
       'ML: SSubU32=3 SCmpEQI32=1 (expected 1 of each)']
  ```
  Root cause: the test was counting raw SSubU32 across all categories
  rather than filtering to `category="LCC"`. The other 2 SSubU32 per
  body are ShadowLimitA/B decrements emitted by `globalReadIncrement*`
  (correctly tagged GRIncA/GRIncB). Test refined to filter; now passes
  with exactly 1 LCC SSubU32 + 1 LCC SCmpEQI32 per main-loop body.

### Cycle 2 surprise — GRA/GRB OrderInverted residuals

Per the bead's "If a Cycle's GREEN reveals a residual divergence ...
STOP and surface" rule, Cycle 2 surfaced an unanticipated finding:
3 `OrderInvertedFailure`s on GR (global-read) nodes in the ML body.

Mechanism: the per-tile schedule on the CMS side mutates kernel-level
flags (`UsePLRPack=True`, `UseMFMAF32XEmulation`) before SIA3 runs,
which changes the GR scheduling order. The SHADOW capture inherited
those mutations because it shared the CMS-mutated `kernel` dict; the
non-CMS reference build (Approach A) uses an unmutated `kernel` dict
per `2LZD_INVESTIGATION.md §6.2 Q2` (the "two builds, accept whatever
Tensilelite mutates" framing). So the GR-stream divergence is the
*expected* Q2 surfacing.

Specific failures (pinned in
`test_non_cms_reference_compare_graphs_surfaces_only_known_residuals`):

- `OrderInvertedFailure(producer=GRA[4], consumer=GRA[3], body=ML, …)`
- `OrderInvertedFailure(producer=GRB[0], consumer=GRA[6], body=ML, …)`
- `OrderInvertedFailure(producer=GRA[6], consumer=GRA[5], body=ML, …)`

All within the same ML body, all GR-side, all `iter_delta=0`.

This is **in scope for `rocm-libraries-3ija`** per the bead's
verification checklist ("If the second-build approach surfaces a
comparison failure that wasn't in the shadow path's surface, STOP and
surface — don't try to fix it inline; that's likely 3ija scope.").

### d3zj closure (LCC capture)

`D3ZJ_SCMPEQI32_INVESTIGATION.md §3.4` identified the SHADOW capture's
LCC absence in ML/ML-1 as a capture-timing defect: `builder.finalize()`
ran at `KernelWriter.py:4591` BEFORE the `closeLoop` emission, so LCC
never landed in the captured body.

nyb5's helper closes this defect at the helper level: the non-CMS
build naturally emits `closeLoop` at `KernelWriter.py:4794` (the
`if not skipClose and not kernel["UseCustomMainLoopSchedule"]:` branch),
and `_appendCloseLoopLCCToBuilder` walks the returned module to
append LCC instructions to the builder before finalize. Verified by
`test_non_cms_reference_has_lcc_in_every_main_loop_body`.

### d3zj xfails — deferred, not removed (rationale)

The two strict-xfailed d3zj tests in
`test_dataflow_graph_emission_ordinal.py` (lines 403-457 and 460-514)
remain xfail in this bead. Removing the xfail markers would require
re-routing those tests from the SHADOW capture
(`writer._last_default_capture` on a CMS-mode build) to the new
helper's Build #2 capture.

Two reasons NOT to do that re-routing in this bead:

1. **The d3zj tests use the broader per-render-counts /
   per-ordinal-class invariants**, not just LCC. Per
   `2LZD_INVESTIGATION.md §6.2 Q2`, the two builds are allowed to
   diverge on whatever Tensilelite mutates internally — the per-tile
   schedule's `UsePLRPack=True` flip on the CMS side moves pack
   instructions out of ML. The d3zj per-render-counts test would FAIL
   under Approach A on this bigger divergence (61 mismatches observed
   when I prototyped the re-route), not because of any bug, but because
   the test's invariant is too broad for the cross-build comparison.
2. **`compare_graphs` body-label-tolerance work (oram.1) is
   prerequisite** for the broader cross-build comparison to behave
   correctly (per `PRELOOP_CAPTURE_PHASE1.md §7`). Until oram.1 lands,
   re-routing d3zj tests to Approach A would surface body-label-
   sensitivity false positives on top of the legitimate Q2 divergence.

The follow-up bead under `rocm-libraries-71hw` should:
- Land oram.1 + z012.
- Re-route the d3zj tests to consume `build_non_cms_reference` for the
  default side.
- Narrow the d3zj invariants to the LCC-specific subset (per
  `D3ZJ_SCMPEQI32_INVESTIGATION.md §2`'s "ML-1 and ML must each have
  exactly one LCC pair; NGL, NLL, PRO, POST_LOOP have zero" wording),
  OR drop the broader invariant entirely once Q2 implementation-detail
  reading is fully baked in.

The d3zj LCC closure ITSELF is verified at the helper level by
`test_non_cms_reference_has_lcc_in_every_main_loop_body` in this bead.

### Full unit suite

`pytest --ignore=…/test_MatrixInstructionConversion.py`:

```
1027 passed, 3 skipped, 4 xfailed in 16.47s
```

Baseline was 1024 passed, 3 skipped, 3 xfailed (pre-nyb5). Delta:
- +3 passed (the 3 new GREEN tests in
  `test_approach_a_non_cms_reference.py`).
- +1 xfailed (Cycle 2 strict-XFAIL pinning the GR-OrderInverted
  residual).
- 0 failed, 0 xpassed.

The 4 xfails are: 2 d3zj LCC (deferred; see above), 1 pre-existing
`test_filterLogicFilesByPredicates_match_emulation_ids` (unrelated),
1 new Cycle 2 strict-XFAIL.

### Shadow path untouched (verification)

`grep -n "_captureDefaultSchedule" KernelWriter.py` confirms all 9
original flag-check sites (3858, 4540, 4634, 4728, 4755, 5142, 5242,
5300, 5458) remain unchanged. The auto-activation site at line 4918
(originally 4717-4718, shifted by additions above it) remains
unchanged in body. `git diff` over `KernelWriter.py` shows only
ADDITIVE hunks: a new public method, a new branch within the non-CMS
arm of `_loopBody`, a new post-closeLoop hook, a new
`_appendCloseLoopLCCToBuilder` helper, a new noLoadLoop non-CMS
capture branch, a new `kernelBody` parallel-assembly branch.

The 1 small refactor in `_loopBody` (lines 4568-4632) widens the
non-CMS branch from a single-line call to an if/else — the `else`
clause is byte-identical to the previous non-CMS line. No behavior
change for callers without the new flag.

## Cross-references

- `2LZD_INVESTIGATION.md §6` — Approach A pick (2026-05-12).
- `2LZD_INVESTIGATION.md §6.2` — Q2/Q3/Q5 follow-up decisions.
- `PRELOOP_CAPTURE_PHASE1.md §7` — body-label-tolerance is critical-path
  for any cross-build comparison; oram.1 is prerequisite for the
  broader Approach-A integration.
- `D3ZJ_SCMPEQI32_INVESTIGATION.md` — LCC capture-timing defect that
  this bead closes at the helper level.
- `D3ZJ_NGL_NLL_LCC_INVESTIGATION.md` — per-body LCC invariant scope
  (ML-1 and ML, not NGL/NLL).
- `rocm-libraries-71hw` — the meta-bead under which this work is
  decomposed.
- `rocm-libraries-czby` — owns SHADOW path deletion (separate bead;
  out of scope for nyb5).
- `rocm-libraries-78n3` — owns `cms_from_default` migration to the
  same `build_non_cms_reference` pattern (separate bead; out of scope
  for nyb5).
- `rocm-libraries-aixt` — owns the `test_prologue_capture.py`
  body-collapse test architecture rework (separate bead; out of
  scope for nyb5).
- `rocm-libraries-3ija` — owns the GR-OrderInverted residual surfaced
  in Cycle 2 (separate bead; out of scope for nyb5).
