# hxcx Implementation Plan — Fix `build_prologue_capture` capture-pipeline ordering

**Bead:** `rocm-libraries-hxcx`
**Branch:** validator_long_term_plans

---

## §1 Scope

`build_prologue_capture` (`ScheduleCapture.py:2336`) currently assembles the
prologue body's `instructions` list by appending all PackA leaves for all
plrIdx first, then all PackB leaves — in pre-interleave, side-monolithic order.
This discards the B-interleave and the `SNop` pads that `_interleavePackAB`
inserts, causing `cumulative_issue_cycles` to see 0 intervening instructions
between MFMA and CVT_PACK pairs that in reality have 7+ quad-cycles of
separation, producing 24 spurious `TimingTooCloseFailure` entries on the
canonical TF32 4x4 TN kernel. The fix moves the capture point to
`KernelWriter.py:5947` — immediately after `_interleavePackAB` populates
`packPrePrefetchItems` — and replaces the per-side `prologue_prefetch_pack_a/b`
Module snapshots with a single post-interleave ordered list carrying per-leaf
side tags, so `build_prologue_capture` receives a single source-of-truth
that already preserves emission order including SNops.

---

## §2 Investigation Findings

### A — `packPrePrefetchItems` shape and context

`packPrePrefetchItems` (KernelWriter.py:5946) is a plain Python `list` of raw
rocisa leaf objects. `_interleavePackAB` appends into it: interleaved A and B
leaves from `packPrePrefetchA.flatitems()` / `packPrePrefetchB.flatitems()`,
plus freshly-constructed `SNop(waitState=…)` objects inserted between the
MFMA-pair and the following CVT_PACK group (KernelWriter.py:886, 920, 941).

`prologue_prefetch_pack_a[plrIdx]` and `..._pack_b[plrIdx]` are rocisa
`Module` objects (`_Module_for_prologue_snap()`) populated at lines 5955–5958
by re-walking `packPrePrefetchA.flatitems()` / `packPrePrefetchB.flatitems()`
**after** `_interleavePackAB` has already consumed and reordered those same
iterators. These per-side snapshots therefore contain exactly the pre-interleave
A-only and B-only leaves in their original uninterleaved order — they never see
each other's leaves and never see the SNop pads.

`_interleavePackAB` is called once per `plrIdx` iteration of the prefetch-local
`for plrIdx in range(0, self.states.numItersPLR)` loop. The snapshot must land
inside that same loop, after the call at line 5947, to capture per-plrIdx
leaves with the correct plrIdx tag.

### B — `build_prologue_capture` end-to-end

Lines 2398–2403 (verified current):

```python
if prefetch_pack_a is not None:
    for plr_idx, mod in enumerate(prefetch_pack_a):
        _append_module(mod, f"PackA{plr_idx}")
if prefetch_pack_b is not None:
    for plr_idx, mod in enumerate(prefetch_pack_b):
        _append_module(mod, f"PackB{plr_idx}")
```

`_append_module` (lines 2373–2396) calls `module.flatitems()`, skips
`TextBlock` and `Label`, and calls `builder.append(inst=leaf, category=…,
subiter=0, slot_kind=SLOT_KIND_PRE_LOOP, mfma_index=-1)` for each leaf.

The validator gets `slot.mfma_index` and `slot.sequence` from
`TaggedInstruction.slot` (a `SchedulePosition`), and `category` from
`TaggedInstruction.category`. `cumulative_issue_cycles` iterates the prologue
body's `instructions` list linearly — it is position-count-based, not
identity-based. So the order of items in `instructions` IS the order the
simulator walks.

SNop pads: `_NoDataflowRule` in the wiring extractor returns empty read/write
sets for `SNop`, so SNop contributes 0 dataflow edges but does contribute to
the timing walk's instruction-position counter — exactly what we need. The
current `_append_module` filter does NOT skip SNop (only TextBlock/Label), so
SNops that reach the builder go in correctly. The problem is they never reach
the builder because they are only in `packPrePrefetchItems`, not in the
per-side Module snapshots.

### C — Tagging requirement

The `category` field (`"PackA{plrIdx}"` or `"PackB{plrIdx}"`) is consumed by:
1. `cms_to_timeline._name_idx_for` — assigns the ordinal `[N]` suffix.
2. `build_dataflow_graph` — partitions leaves into A vs B streams for dataflow
   edge keys.
3. The validator failure message (human-readable).

If we snapshot the post-interleave stream without side tags, every leaf would
be in an unknown category and downstream naming/routing would break. The tag is
not merely diagnostic — it is structurally required. The tag cannot be inferred
from the rocisa instruction class alone (both sides use `VCvtPkBF16ToFP32` and
`VMfmaF32_4x4x4_BF16`; register operands differ but parsing them is not
appropriate here). SNop objects are fresh insertions from `_interleavePackAB`
with no origin — they should receive the canonical neutral category `"SNOP"`
(uppercase, matching `_RECOGNIZED_CATEGORY_EXACT` at ScheduleCapture.py:1711
and `_OPTSCHEDULE_IDMAP_SKIP_CATEGORIES` at CMSValidator.py:5063).
`_NoDataflowRule.applies()` uses `_category(inst)` (rocisa class-based), not
the category string, so the string casing does not affect dataflow routing —
but skip-sets and recognized-category checks ARE string-based. Using `"SNop"`
(mixed case) would silently escape those skip-sets. Use `"SNOP"` throughout.

### D — Snapshot mechanism

The surrounding code uses `Module.flatitems()` for leaf iteration (not
`structural_clone` or `deepcopy`). The leaves from `packPrePrefetchA/B` are
the same Python objects that eventually get `module.addItems(packPrePrefetchItems)`
at line 5962. Python identity is preserved through `flatitems()` —
`structural_clone` is only needed when the Module wrapper tree is mutated by a
consumer (SIA3's `popFirstItem`). Here the consumer is `addItems`, which copies
references into the `module` tree but does not mutate the source leaf objects.

The correct snapshot mechanism is:
- Before calling `_interleavePackAB`: build `{id(leaf): "PackA{plrIdx}"}` from
  `packPrePrefetchA.flatitems()` and `{id(leaf): "PackB{plrIdx}"}` from
  `packPrePrefetchB.flatitems()`.
- After `_interleavePackAB` populates `packPrePrefetchItems`: iterate the list
  and look up each item's category by `id(item)` from the pre-built dict.
  SNops (not in the dict) get category `"SNop"`.
- Accumulate `(leaf, category, plrIdx)` tuples into a per-plrIdx list and
  store on `self._capture_context` as the new `prologue_prefetch_items`.

### E — The 16 `EdgeRoutedDifferentlyFailure` entries

All 16 failures (verified in `compare_graphs_failures.txt`) involve
PackA3/PackB3 **mainloop** consumers routing to prologue PackA0/PackB0
producers at `idx=-1`. The reference graph finds no prior writer at the
consumer position (or finds a different prologue-side writer) because the
prologue's ordinal numbering is wrong — with the current misordered capture,
`PackA0[1]`, `PackA0[3]`, `PackA0[9]` etc. map to physically different
instructions than they do in the subject graph. After the fix re-orders the
prologue capture, the ordinal assignments will shift and the reference-graph
prologue producers will re-map to their correct physical leaves, resolving the
routing disagreement.

Confidence: HIGH — all 16 failures reference `idx=-1` (prologue) producers,
and the failure pattern (`reference routes through PackA0[N] @ idx=-1 (of next
iteration)`) directly depends on which leaf gets ordinal N in the prologue
body.

Residual risk (LOW): a small number of the 16 could be sensitive to
subiter-related ordering of the mainloop bodies rather than the prologue
ordinals. If any persist after the fix, they become a separate follow-up bead.

### F — Test impact

Tests that exercise `build_prologue_capture` directly or indirectly:

1. `test_prologue_capture.py` — `test_build_prologue_capture_returns_none_when_all_inputs_empty`:
   calls `build_prologue_capture()` with no args / empty lists. Unaffected —
   the new signature will still accept a single `prologue_interleaved_items`
   argument (or keep backward-compat empty-list path). After the fix this test
   continues to return `None`.

2. `test_prologue_capture.py` — `test_preloop_divergence_catches_useplrpack_change`:
   asserts that `cap_with.prologue` has Pack-tagged instructions. After the fix
   the prologue will have the SAME Pack instructions in interleaved order —
   still non-empty, still Pack-tagged. PASS.

3. `test_prologue_capture.py` — `test_whole_kernel_cms_prologue_matches_non_cms_reference`:
   asserts `len(cms_cap.prologue.instructions) > 0`. Still true. PASS.

4. `test_cross_subiter_alu_carveout_real_kernel.py` —
   `test_real_kernel_validates_clean_with_carveout_engaged`: currently 1 ERROR
   (fixture raises due to `TimingTooCloseFailure` in `validate_edge_wait_coverage`
   being surfaced as an assertion error). After the fix the prologue capture is
   correctly ordered, `validate_edge_wait_coverage` finds no violations, and
   the fixture builds cleanly. The test asserts `compare_graphs == []`. This
   should also hold — the 16 `EdgeRoutedDifferentlyFailure` entries that
   currently appear in `compare_graphs_failures.txt` are expected to resolve.
   Test becomes PASS.

No test pins the current misordered instruction sequence (no test asserts
`instructions[5]` is a specific leaf by position). The structural tests only
check `len > 0` and `category.startswith("Pack")`.

### G — Failure-count delta prediction

Current state: 20 FAILED + 1 ERROR (per revert_C3g plan; confirmed by C4
plan's "20 FAILED + 2 ERROR going into C4" with one ERROR being the carveout
test; the revert plan shows "1 ERROR" = `test_real_kernel_validates_clean_with_carveout_engaged`).

After hxcx:
- The 1 ERROR (`test_real_kernel_validates_clean_with_carveout_engaged`) resolves
  to PASS: prologue timing walk is now correct, `validate_edge_wait_coverage`
  returns no `TimingTooCloseFailure`, fixture does not raise.
- The 16 `EdgeRoutedDifferentlyFailure` entries are in `compare_graphs` output,
  which `test_real_kernel_validates_clean_with_carveout_engaged` asserts == [].
  These resolve if the prologue-ordinal re-mapping holds (HIGH confidence).
- Pre-existing failures (u6nn, nyb5, and any other pre-existing non-C-chain
  beads) remain unchanged.

Predicted: 20 FAILED + 0 ERROR → net change: -1 ERROR, 0 FAILED delta.
If the 16 EdgeRouted entries do NOT fully resolve: still -1 ERROR, still
PASS for the fixtures that don't assert on `compare_graphs`.

### H — Architectural layer check

The fix is in the capture layer (`ScheduleCapture.py` / `KernelWriter.py`
snapshot site), which is the right layer. The capture is the canonical
representation the validator consumes. Fixing it here upholds the
"validate as single timeline" standing rule — the captured timeline must
match the actual emission order. The alternative (teaching the validator to
be aware of pre-interleave A-then-B ordering) would be a validator hack that
hides the mismatch rather than correcting it. Standing-rule check: PASS.

---

## §3 Design

### New snapshot pattern

**Before `_interleavePackAB` call** (inside the `for plrIdx in range(…)` loop,
at KernelWriter.py:5946):

```python
if getattr(self.states, "_captureDefaultSchedule", False):
    _prologue_id_to_category = {
        id(leaf): f"PackA{plrIdx}"
        for leaf in packPrePrefetchA.flatitems()
    }
    _prologue_id_to_category.update({
        id(leaf): f"PackB{plrIdx}"
        for leaf in packPrePrefetchB.flatitems()
    })
```

**After `_interleavePackAB` call** (still inside the same loop body,
at KernelWriter.py:5947):

```python
if getattr(self.states, "_captureDefaultSchedule", False):
    _prologue_snap = [
        (leaf, _prologue_id_to_category.get(id(leaf), "SNOP"))
        for leaf in packPrePrefetchItems
    ]
    self._capture_context.prologue_interleaved_items.extend(_prologue_snap)
```

NOTE: `"SNOP"` (uppercase) is the canonical category string; it matches
`_RECOGNIZED_CATEGORY_EXACT` (ScheduleCapture.py:1711) and
`_OPTSCHEDULE_IDMAP_SKIP_CATEGORIES` (CMSValidator.py:5063). Do NOT use `"SNop"` (mixed case).

The existing `prologue_prefetch_pack_a` / `prologue_prefetch_pack_b` lists on
`CaptureContext` are replaced by a single `prologue_interleaved_items: list`
initialized to `[]` in `CaptureContext.__post_init__` / `reset()`.

### `build_prologue_capture` new signature

```python
def build_prologue_capture(*, prologue_interleaved_items=None):
```

Takes a list of `(leaf, category)` tuples in emission order. The internal
`_append_module` helper is replaced by direct iteration:

```python
for leaf, category in (prologue_interleaved_items or []):
    cls_name = type(leaf).__name__
    if cls_name in ("TextBlock", "Label"):
        continue
    builder.append(
        inst=leaf,
        category=category,
        subiter=0,
        slot_kind=SLOT_KIND_PRE_LOOP,
        mfma_index=-1,
    )
    any_appended = True
```

SNop leaves (category `"SNop"`) pass the filter and enter the builder. The
`_NoDataflowRule` gives them empty read/write sets; their position in the
`instructions` list is what the timing walk needs.

### SNop slot assignment

SNops get `slot_kind=SLOT_KIND_PRE_LOOP`, `mfma_index=-1`, `subiter=0`,
`category="SNop"`. This is consistent with every other prologue leaf. The
timing simulator counts them as 1 instruction position regardless of category.

---

## §4 Step-by-step Implementation Order

1. **Add `prologue_interleaved_items: list` to `CaptureContext`** in
   `ScheduleCapture.py`. Initialize to `[]` in `__post_init__` / `reset()`.
   Remove `prologue_prefetch_pack_a` and `prologue_prefetch_pack_b` fields
   (delete them completely — they are hacks, do not leave them as deprecated
   aliases).

2. **Replace the snapshot block in `KernelWriter.py:5954–5958`** with the
   two-phase id-dict + post-interleave tuple-list pattern from §3. The
   `_captureDefaultSchedule` guard wraps both phases. Remove all references to
   `prologue_prefetch_pack_a` and `prologue_prefetch_pack_b` from KernelWriter.py.

3. **Update `KernelWriter.py:6015–6018`** (the prologue-end checkpoint): call
   `build_prologue_capture(prologue_interleaved_items=ctx_for_prologue.prologue_interleaved_items)`
   instead of the old `prefetch_pack_a=` / `prefetch_pack_b=` arguments.

4. **Rewrite `build_prologue_capture`** in `ScheduleCapture.py` with the new
   signature. Remove the old `_append_module` helper and the two-loop append
   block. Replace with the direct `(leaf, category)` iteration from §3.
   Use `"SNOP"` (uppercase) as the fallback category for unrecognized leaves.

4a. **Update `test_build_prologue_capture_returns_none_when_all_inputs_empty`**
    in `test_prologue_capture.py`: change the second call from
    `build_prologue_capture(prefetch_pack_a=[], prefetch_pack_b=[])` to
    `build_prologue_capture(prologue_interleaved_items=[])`. Both calls must
    still return `None`.

5. **Update `build_prologue_capture` docstring**: replace references to
   `prefetch_pack_a/b` with `prologue_interleaved_items`; document that the
   list carries `(leaf, category)` tuples in emission order including SNOP pads.

6. **Delete initialization of `prologue_prefetch_pack_a/b` at
   `KernelWriter.py:5855–5857`** (the `_Module_for_prologue_snap()` allocations).
   These are no longer needed.

7. **Run targeted test**: `test_prologue_capture.py` and
   `test_cross_subiter_alu_carveout_real_kernel.py`.

8. **Run full unit suite** (excluding `test_MatrixInstructionConversion.py`).
   Confirm: 20 FAILED + 0 ERROR (net change: -1 ERROR).

---

## §5 Validation

**Targeted re-run:**
```
pytest Tensile/Tests/unit/test_cross_subiter_alu_carveout_real_kernel.py \
       Tensile/Tests/unit/test_prologue_capture.py -v \
       --ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py
```

Expected outcome:
- `test_real_kernel_validates_clean_with_carveout_engaged`: PASS (was ERROR).
  `validate_edge_wait_coverage(subj_graph)` returns `[]`; `compare_graphs`
  returns `[]` (all 16 EdgeRouted entries resolved by correct prologue ordinals).
- All `test_prologue_capture.py` tests: PASS (structural assertions unaffected).

**Full suite:**
```
pytest Tensile/Tests/unit/ \
       --ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py -v
```
Expected: 20 FAILED + 0 ERROR.

---

## §6 Re-fixture Work

No test pins the misordered instruction sequence. However, two tests require
updates to their call sites because the old `build_prologue_capture` keyword
arguments (`prefetch_pack_a=`, `prefetch_pack_b=`) are being removed:

1. **`test_build_prologue_capture_returns_none_when_all_inputs_empty`**: This
   test calls `build_prologue_capture(prefetch_pack_a=[], prefetch_pack_b=[])`.
   With the new signature `build_prologue_capture(*, prologue_interleaved_items=None)`,
   those keyword args become `TypeError`. Update the test to call
   `build_prologue_capture()` and `build_prologue_capture(prologue_interleaved_items=[])`
   instead. Both must still return `None`.

2. **`test_preloop_divergence_catches_useplrpack_change`** docstring: mentions
   `ctx.prologue_prefetch_pack_a/b` as the storage mechanism — update to
   reference `prologue_interleaved_items` and the new call site. The test
   assertions themselves do not reference the old field names and are unaffected.

---

## §7 Risks / Open Questions

1. **Multiple plrIdx accumulation**: The fix extends `prologue_interleaved_items`
   across the `for plrIdx in range(numItersPLR)` loop. With PGR=2 and
   `numItersPLR=2`, this correctly produces one flat ordered list containing
   both plrIdx=0 and plrIdx=1 interleaved segments in emission order (since
   `_interleavePackAB` is called once per `plrIdx`). The validator walk over the
   full list correctly represents the physical prologue as one linear stream.
   Low risk — the loop is sequential and `packPrePrefetchItems` is reset to `[]`
   at the start of each plrIdx iteration.

2. **16 EdgeRoutedDifferentlyFailures**: Assessed HIGH-confidence to resolve.
   If any persist, they are a separate follow-up bead and do not block the hxcx
   commit. The 1 ERROR → PASS is the hard acceptance gate.

3. **SNOP category string**: Use `"SNOP"` (uppercase) for injected nop pads.
   This is already the canonical string in `_RECOGNIZED_CATEGORY_EXACT`
   (ScheduleCapture.py:1711) and `_OPTSCHEDULE_IDMAP_SKIP_CATEGORIES`
   (CMSValidator.py:5063). `_NoDataflowRule.applies()` is class-based (not
   string-based) so it is unaffected by case, but the skip-set lookups are
   exact-string and would miss `"SNop"`. RESOLVED: always use `"SNOP"`.

4. **plrIdx tag on SNOP**: SNops are inserted by `_interleavePackAB` between
   groups from both sides — assigning them `"SNOP"` rather than forcing them
   into a specific plrIdx is correct, since they are not plrIdx-specific
   instructions.

---

## §8 New Beads to File

None generated by this investigation. All discovered risks are residuals of
pre-existing beads (u6nn for `test_prologue_capture.py` breakage, 6jbr for
LR/LW/GR prologue capture). No new structural defects surfaced.
