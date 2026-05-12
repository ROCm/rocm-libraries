# rocm-libraries-vybd (F3) — Implementation memo

## What F3 does

F3 deletes the default-side leftover-pack walk in `KernelWriter.py`'s
`_loopBody` method (the block under
`if getattr(self.states, "_captureDefaultSchedule", False):` that previously
ran after the per-iter capture loop and before `customMainLoopSchedule`).

**Removed (lines previously at KernelWriter.py:4574-4628):**
- Construction of a fresh `leftover_idmap` over the canonical pack-code
  sources via `build_idmap(...)`.
- `invert_idmap_to_id_to_category(...)` to derive `leftover_id_to_cat`.
- Prefetch-pack leaf re-tagging into `leftover_id_to_cat` for both A-side
  and B-side prefetch buffers.
- The double `for buf in list(pack) + list(packPre):` walk that appended
  every leaf still in those storage buffers to the capture builder as
  `slot_kind=SLOT_KIND_POST_LOOP, mfma_index=-1` entries.

**Preserved (lines previously at KernelWriter.py:4629-4630):**
- `self._capture_context.default_main = builder.finalize()`
- `self._capture_context.builder = None`

The `default_main` handoff is **still required**: it is consumed downstream
in `kernelBody` (see `KernelWriter.py:5274`, `main = ctx.default_main`)
to assemble the `FourPartCapture` that the cross-scheduler validation
operates on. Both the A-side `pack[*]` walk and the B-side `packPre[*]`
walk go together — there is no half-state.

The post-deletion block is a 4-line capture-completion handoff with no
leftover-walk noise:

```python
if getattr(self.states, "_captureDefaultSchedule", False):
    # ... (multi-line architectural comment naming vybd, xbi0, flpk, 71hw) ...
    builder = self._capture_context.builder
    if builder is not None:
        self._capture_context.default_main = builder.finalize()
        self._capture_context.builder = None
```

## Why F3 supersedes xbi0 and flpk

Both prior bead investigations isolated defects whose **host code** was
the leftover-pack walk:

- **rocm-libraries-xbi0 (Approach X)** patched the walk to dedup leaves
  whose `id(leaf)` was already in the per-iter capture's `seen` set.
  This was a tactical fix inside the walk that left the architectural
  scaffolding in place.

- **rocm-libraries-flpk** identified that even with xbi0's same-id
  dedup, the walk still emitted *distinct* Python objects with
  *identical canonical text* under different `PackA{u}` category tags
  (PackA0 from per-iter PRE_LOOP, PackA3 from leftover POST_LOOP),
  which would surface as cross-graph node identity divergence under the
  4up4 per-emission-ordinal identity scheme.

F3 deletes the walk outright. The host code for both defects is gone, so
neither defect can manifest from this site again.

## Architectural alignment with rocm-libraries-71hw Approach A

The walk's stated rationale (per the long comment block at the original
lines 4574-4584) was:

> CMS aggregates ALL iters' pack content into its main_loop macro, so
> for a comparable framing the shadow's main_loop must also include the
> leftover.

This rationale assumed a **shadow capture** model: the default-side
build was a synthetic mirror constructed alongside the real CMS-driven
emission, intended to be byte-comparable to the CMS macro shape rather
than to be itself a runnable kernel.

Under **rocm-libraries-71hw Approach A** the default-side build is
instead a **true non-CMS reference kernel** (`UseCustomMainLoopSchedule=0`
running through the natural emission path). Both sides of `compare_graphs`
are real, emittable, runnable kernels with naturally-emitted main-loop
bodies. There is no shadow. The leftover-pack framing requirement
disappears, and the walk becomes dead scaffolding whose only remaining
effect is to inject mistagged leaves into the capture.

F3 is the architectural fix that aligns the capture pipeline with
Approach A.

## Verification

### On `users/alvasile/validator_long_term_plans` (this branch's base)

Validation-branch baseline (without F3):
```
1000 passed, 3 skipped, 1 xfailed
```

With F3 deletion alone (no test changes):
```
1000 passed, 3 skipped, 1 xfailed
```

With F3 deletion + 5 new invariant tests in
`test_capture_pipeline_checks.py`:
```
1005 passed, 3 skipped, 1 xfailed
```

**Net: zero validation-branch tests broken**, confirming the FLPK
investigator's prediction. The test count grows by exactly the 5 newly
added tests:

- `TestNoDoubleCaptureUnit::test_builder_with_no_aliased_leaves_passes`
- `TestNoDoubleCaptureUnit::test_assertion_fires_when_same_inst_appended_twice`
- `TestNoCanonicalTextCrossTaggedUnit::test_builder_with_distinct_canonical_texts_passes`
- `TestNoCanonicalTextCrossTaggedUnit::test_same_canonical_text_same_category_passes`
- `TestNoCanonicalTextCrossTaggedUnit::test_assertion_fires_on_canonical_text_cross_tagging`

Pytest invocation used:
```
python -m pytest Tensile/Tests/unit/ \
  --ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py \
  --timeout=120 -q
```
(Excluding the slow `test_MatrixInstructionConversion.py` per the
project-wide convention.)

### On the 4up4 branch (measured, F3 cherry-picked into a SCRATCH worktree
of `worktree-agent-a4185c2af90971c4a`)

**4up4 baseline (no F3):**
```
15 failed, 981 passed, 3 skipped, 4 xfailed, 2 errors
```
(The 15 failed + 2 errored = the 17 previously-failing tests cluster.)

**4up4 with F3 layered:**
```
1 failed, 997 passed, 3 skipped, 3 xfailed, 1 xpassed
```

**16 of 17** previously-failing tests now pass:
- All 10 `test_ScheduleCapture.py` failures fixed
  (`TestRealKernelCapture::test_tf32_4x4_tn_capture_shape`,
   `TestRealKernelCapture::test_tf32_4x4_tn_capture_categories`,
   `TestPhase4DefaultCapture::test_default_capture_populated`,
   `TestPhase4DefaultCapture::test_default_and_cms_captures_both_populated`,
   `TestPhase5DefaultTailCapture::test_n_gl_and_n_ll_populated`,
   `TestPhase5DefaultTailCapture::test_no_false_positive_on_clean_cms_kernel`,
   `TestPhase5DefaultTailCapture::test_n_gl_n_ll_state_resets_after_kernel`,
   `TestDataflowGraphIntegration::test_dataflow_gating_passes_on_clean_cms_kernel`,
   `TestDataflowGraphIntegration::test_dataflow_gating_passes_with_MIArchVgpr_true`,
   `TestPgrPlrCaptureMatrixEndToEnd::test_pgr2_snll_false_matches_baseline`)
- All 4 `test_cms_from_default.py` failures fixed
- Both 2 `test_cross_subiter_alu_carveout_real_kernel.py` errors fixed

**Residual failure (NOT a F3 regression):**
- `test_prologue_capture.py::test_whole_kernel_useplrpack_cms_matches_both_defaults`
  still fails with `CaptureConsistencyError: data-flow node identity sets
  differ. in subject but not reference: 16 identities ({'MFMA': 16})`.
  This is a **whole-kernel MFMA-accounting divergence** between CMS and
  default — independent of the leftover-pack walk. Verified by running
  the same test on the 4up4 baseline (pre-F3): it failed identically
  there too. Out of scope for vybd F3.

**Of the 3 `xfail`-marked regressions in
`test_dataflow_graph_emission_ordinal.py:376-471`:**
- `test_example_yaml_no_spurious_order_inverted_failures`
  → **flipped to XPASS** (F3 fixed it). Removing this test's `xfail`
  marker is a separate 4up4-side fixup.
- `test_real_kernel_per_render_counts_match` → **still XFAIL**.
  Force-running it shows residual mismatches in SYNC/control instruction
  counts (`s_waitcnt vmcnt(N)`, `s_nop 0`, `s_barrier`,
  `s_sub_u32 sgprLoopCounterL,1`, `s_cmp_eq_i32 sgprLoopCounterL`) —
  not pack-related. This is a deeper CMS-vs-default whole-kernel
  divergence that F3 does not address. The xfail marker should remain.
- `test_real_kernel_per_ordinal_logical_instruction_matches` → still XFAIL
  (depends on the per-render-count invariant above).

**Summary on 4up4:** F3 closes ~94% of the targeted regression cluster
(16/17 + 1/3 xfail flip). The two residual issues (whole-kernel MFMA
accounting + control-instruction count divergence) are independent
defects with their own scope.

### Strengthened invariant tests

The new tests in `test_capture_pipeline_checks.py` pin both:

1. **xbi0's same-id invariant** (`_assert_no_double_capture_in_body`):
   no two `TaggedInstruction`s in a single body share the same
   `id(rocisa_inst)`.

2. **flpk's canonical-text invariant**
   (`_assert_no_canonical_text_cross_tagged_in_body`): no two
   `TaggedInstruction`s in a single body share the same
   `WrappedInstruction.canonical_str(...)` under DIFFERENT category tags.
   This is **strictly stronger** than the same-id invariant — flpk's
   pairs have different ids but identical canonical text, so the
   same-id check would not catch them.

The two invariants are mechanism-agnostic capture-time properties of
`LoopBodyCapture.instructions`. They are independent of the identity
scheme used downstream in `compare_graphs` (historical class_tag-based
or current per-emission-ordinal), so they pin the capture-pipeline
behaviour against future regression regardless of which downstream
identity scheme is in play.

The test for `same_canonical_text_same_category_passes` documents that
SAME canonical text under the SAME category tag is allowed (legitimate
repeats inside one emission group, e.g. SYNC `s_waitcnt(0)`); the
invariant fires only on cross-tagging.

### No production-kernel behaviour change

F3 only deletes capture-pipeline code that runs under
`self.states._captureDefaultSchedule`. Production kernel emission
(when capture is not enabled) is untouched: the deleted block was
gated on the capture flag, and the rest of `_loopBody` after the
deleted block (`customMainLoopSchedule(...)`, `module.add(...)`, etc.)
is unchanged.

## Reference

- `XBI0_CAPTURE_DEFECT_INVESTIGATION.md` — original defect investigation
  for the xbi0 same-id collision shape; identifies the leftover-pack
  walk as the host code.
- `XBI0_IMPLEMENTATION.md` (discarded with the X branch) — the Approach X
  fix inside the walk; superseded by F3 deleting the walk. Not present in
  this tree; preserved only in the abandoned X-branch git history.
- `FLPK_CROSS_TAGGING_INVESTIGATION.md` — extends the same-id analysis
  to the canonical-text-collision shape; recommends mechanism-agnostic
  pinning at `body.instructions` level (§6); confirms zero
  validation-branch tests pin the existing PackA0/PackA3 cross-tag
  split (§5).
- `EMISSION_ORDINAL_DESIGN.md` §2.7-§2.8 — identity scheme that surfaces
  the cross-tagging as a divergence in `compare_graphs`.
- `2LZD_INVESTIGATION.md` §6 — Approach A decision (true non-CMS
  reference build via `UseCustomMainLoopSchedule=0`); the architectural
  framing under which the leftover walk becomes dead scaffolding.
