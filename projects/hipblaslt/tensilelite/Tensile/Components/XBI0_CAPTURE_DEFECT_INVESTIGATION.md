# XBI0 — Default-Side Pack-MFMA Double-Capture Investigation

Bead: `rocm-libraries-xbi0`. Investigation only — no production code changes.

## TL;DR

- Mechanism: **(a) slot-list double-append**. The default-side leftover-pack
  capture path at `KernelWriter.py:4613-4628` walks `pack[*]` and `packPre[*]`
  arrays at end of `_loopBody`'s for-uIdx loop and appends each leaf via
  `builder.append(...)`. Because the kernel writer at `KernelWriter.py:4211`
  (and four sibling sites) deliberately adds the SAME `packCodeA` Python
  object to **two** `pack[*]` slots when `ForceUnrollSubIter` is active —
  once into `pack[packStoreIdx*N]` and once into `pack[1]` — the leftover
  walk encounters the same leaf twice and emits two `TaggedInstruction`s
  with **the same** category tag (`PackA3` for the canonical TF32 4×4 TN
  kernel) at two different `slot.sequence` positions in the same body.
- The brief's framing of "different category tags (PackA0 + PackA3)" is
  **wrong for the post-loop double-capture observed in the failing tests**.
  The forensic capture run produces two appends BOTH tagged `PackA3` (and
  for B: BOTH tagged `PackB3`). The 2LZD memo's PackA0+PackA3 framing
  describes a separate residual that is downstream of the same `pack[]`
  aliasing convention but operates on the per-iter capture, not the
  leftover walk.
- The principled fix has architectural choices. **The primary recommendation
  needs the user's input on (X) restructure-leftover-walk vs (Y) eliminate-
  pack[1]-aliasing-in-kernel-writer.** Both are at-source fixes; neither
  is tactical.

---

## §1 — Mechanism (Q1, Q2, Q3)

### Q1 — Where the same Python pack-MFMA object is appended twice

**The double-append site is `KernelWriter.py:4613-4628`** (the `_loopBody`
leftover-pack capture, run AFTER the for-uIdx loop has finished consuming
the per-iter pack chain into iterCode):

```python
for buf in list(pack) + list(packPre):
    if buf is None:
        continue
    for leaf in buf.flatitems():
        cat = leftover_id_to_cat.get(id(leaf))
        if cat is None:
            continue
        builder.append(
            inst=leaf, category=cat,
            subiter=kernel["LoopIters"],
            slot_kind=SLOT_KIND_POST_LOOP, mfma_index=-1,
        )
```

The walk visits 8 modules in fixed order: `pack[0..3]` then `packPre[0..3]`.
For each non-empty buf it emits each leaf via `builder.append`.

**The reason a leaf shows up in two of those 8 bufs** is the kernel writer's
`pack[1].add(packCodeA)` aliasing convention at `KernelWriter.py:4211`:

```python
if kernel["UseCustomMainLoopSchedule"]:
    LRCodeAAllIters[uIdx].add(localReadCodeA)
    PackCodeAAllIters[uIdx].add(packPreA)
    PackCodeAAllIters[uIdx].add(packCodeA)
if kernel["ForceUnrollSubIter"] and not self.states.doFullPackCodePrefetch:
    pack[1].add(packCodeA)            # <-- same packCodeA also into pack[1]
```

The same Python `packCodeA` Module is added to BOTH the per-iter primary
container (`pack[packStoreIdx * numIterPerCoalescedReadA]`, line 4205) AND
to `pack[1]` (line 4211). Five sibling sites mirror this: `KernelWriter.py:4223`,
`4235` for A-side metadata / B-side metadata in `_loopBody`, plus
`KernelWriter.py:3395`, `3407`, `3415`, `3427` in `_noLoadLoopBodyDefault`.

The aliasing is intentional and predates schedule capture: `pack[1]` serves
as a SubIter-mode "swap" buffer that aggregates the iter's pack code for
consumption by the next subiter. Kernel emission consumes `pack[packIdx]`
once via SIA3, but the SECOND reference into `pack[1]` survives if pack[1]
is not consumed before the loop ends.

### Q2 — Mechanism classification: **(a) slot-list double-append**

The **same Python object** (verified by `id()`) is appended **twice** to
the same `LoopBodyCaptureBuilder._instructions` list, with the **same**
`(slot_kind, mfma_index)` key, getting consecutive sequence numbers in
that bucket.

This is NOT (b) "category recategorization" — both appends carry the same
category (e.g. both `PackA3`). It is NOT (c) "module aliasing" in the
sense the brief described — there's no second wrapper around the same
underlying instruction; there are two physically distinct
`TaggedInstruction(WrappedInstruction(inst), ...)` objects each pointing
back to the same `inst`.

**Empirical evidence** (forensic run on canonical TF32 4×4 TN config,
`UseCustomMainLoopSchedule=1`, `ForceUnrollSubIter=True` auto-enabled by
`Solution.py:1740`):

```
Builder leftover-walk pack/packPre snapshot at the `for buf in ...` site:
  pack[0]: 40 leaves, 40 unique  (PackA3 + PackB3 from u=3 packStoreIdx=0)
  pack[1]: 20 leaves, 20 unique  (SAME PackA3 leaves as pack[0])
  pack[2]: EMPTY
  pack[3]: EMPTY
  packPre[0..3]: EMPTY
  CROSS-BUF DUP: pack[0] ∩ pack[1]: 20 shared ids
```

Resulting builder appends in `(POST_LOOP, -1)` bucket (60 total, sequences
0..59):

```
seq 0..19   cat=PackA3  (pack[0] flatitems, A-side)
seq 20..39  cat=PackB3  (pack[0] flatitems, B-side)
seq 40..59  cat=PackA3  (pack[1] flatitems, SAME ids as seq 0..19)
```

Per-id confirmation:

```
id=...428432 cat=PackA3 cls=VCvtPkF32toBF16 at seq 0  AND seq 40
id=...570288 cat=PackA3 cls=MFMAInstruction at seq 4  AND seq 44
... (20 such ids)
```

The MFMAInstruction at seq 4 / seq 44 is the same TF32-emulation pack-MFMA
referenced in the brief (the rendering matches `v_mfma_f32_4x4x4_16b_bf16
... // Calculate low bits for TF32 emulation`). The brief's id /
sequence numbers from the prior agent's run are this same artifact.

### Q3 — Why CMS-side captures only once

The CMS-side capture path drives off the `idMap` returned by
`build_idmap` (`Components/CustomSchedule/dispatch.py:136`), keyed by the
`PackCodeAAllIters[uIdx]` list. Each leaf appears in exactly one
`PackCodeAAllIters[uIdx]` entry — the `pack[1].add(packCodeA)` aliasing at
KernelWriter.py:4211 does NOT also push into `PackCodeAAllIters`. The
CMS macro walker (`expand_cms_macro` /
`Components/CustomSchedule/dispatch.py:200-...`) consumes each
`idMap[key]` instruction exactly once when its scheduled `mfmaIndex` is
hit, so each leaf becomes exactly one CMS-side `TaggedInstruction`.

We verified this empirically by hooking `build_idmap`: the leftover-stage
idMap has `{PackA0: 20 leaves, PackA3: 20 leaves, PackB0: 20 leaves,
PackB3: 20 leaves}` with **zero cross-category id duplications and zero
within-category duplications**. The aliasing is contained inside `pack[]`,
not in the per-iter `PackCodeAAllIters[]` lists.

The **asymmetry** is therefore: CMS-side reads from a per-uIdx-tagged
source-of-truth (`PackCodeAAllIters`), default-side leftover walk reads
from the ALIASED storage buffer (`pack[]`). The kernel writer's pack[1]
aliasing is invisible to the per-uIdx tagging path but visible to the
flat buf-walk.

---

## §2 — Recommended fix approach (Q4) — needs user decision

Both candidates below address the SOURCE of the double-capture. Neither
introduces CMS-coupled categorization. Neither is the tactical "dedupe in
`_make_node`" or "skip pack from data-flow filter" the brief rules out.
There IS a meaningful architectural trade-off and the user should weigh in
before implementation begins.

### Option X — Restructure the leftover walk to walk a SET, not the buffers

**Where.** `KernelWriter.py:4571-4630` (the `_captureDefaultSchedule`
leftover branch in `_loopBody`).

**Invariant established.** "The leftover-pack capture appends each Python
leaf at most once per body, regardless of how many `pack[*]`/`packPre[*]`
slots reference that leaf." The walk computes the set of `id(leaf)` values
already appended to the builder during the for-uIdx loop, then emits only
the leaves NOT in that set, ONCE each, in a deterministic order
derived from `PackCodeAAllIters[u]` traversal (uIdx-ascending) for the
PackA*/PackB* leaves, and from a separate well-defined source for any
other categories that legitimately appear in leftover.

**CMS coupling.** None. The seen-set is built from the builder's own
`_instructions`. The per-uIdx ordering source `PackCodeAAllIters` is the
same source `build_idmap` uses for both paths — it is not a
CMS-concept import.

**Trade-offs.**
- Pro: corrects the symptom AND the conceptual mistake (the leftover
  walk's intent is "items not yet captured", but it's literally walking
  storage buffers that may alias).
- Pro: fix is small (the leftover branch is ~50 lines today; the
  rewrite is similar size and stays in `_loopBody`'s capture branch).
- Con: implies the leftover walk now needs to know which leaves the
  per-iter capture ALREADY appended. That can be done by inspecting
  `self._capture_context.builder._instructions` (introducing an
  internal-state read) OR by the per-iter capture itself recording an
  id-set on `self._capture_context`.
- Con: the leftover walk's "what should land in POST_LOOP" question is
  now algorithmic, not "everything still in `pack[]`". If a future kernel
  writer adds a NEW path that emits items into `pack[*]` but not into
  `PackCodeAAllIters`, those items won't make it into POST_LOOP. (Today
  this would be a bug; the rewrite needs a clear contract documented.)

### Option Y — Eliminate the `pack[1].add(...)` aliasing in the kernel writer

**Where.** `KernelWriter.py:3395, 3407, 3415, 3427` (in
`_noLoadLoopBodyDefault`'s SubIter branch) and
`KernelWriter.py:4211, 4223, 4235` (in `_loopBody`'s SubIter branch).

**Invariant established.** "Every Python leaf in the kernel writer's
pack/packPre arrays appears in at most one `pack[*]` and at most one
`packPre[*]` slot at any time." The `pack[1].add(packCodeA)` lines are
removed; the SubIter-mode "swap" pack chain is reconstructed at
consumption time (or via a dedicated separate `subIterPack` Module
populated from a structurally cloned copy of `packCodeA`).

**CMS coupling.** None. The change lives entirely in the kernel writer's
buffer-management; the schedule-capture code is untouched.

**Trade-offs.**
- Pro: the underlying invariant ("a leaf belongs to one position in the
  storage tree") is restored. The capture path's leftover walk then
  reads from a coherent storage tree without needing dedup logic.
- Pro: the same fix benefits any future capture / introspection / debug
  tooling that walks `pack[]` — they all currently have the same
  aliasing hazard.
- **Con (the big one): this is a kernel-EMISSION change.** The
  `pack[1]` aliasing is consumed by the SubIter scheduling logic in
  ways I have not exhaustively traced. Removing it without
  understanding the full consumer set risks producing different
  generated assembly for `ForceUnrollSubIter` kernels — a regression
  surface much larger than the capture pipeline.
- Con: pre-dates the capture work; the convention ships in production
  kernels TODAY. Even if it's structurally suspect, treating it as a
  bug rather than a contract requires more evidence than this
  investigation collected.

### Why this needs user input

Option X is **the correct fix for the capture pipeline as scoped by the
bead** ("the bug must be fixed at the SOURCE of the double-capture — in
the KernelWriter capture pipeline itself"). It treats the kernel writer's
`pack[1]` aliasing as a precondition the capture pipeline must adapt to,
not as a defect.

Option Y is **the correct fix for the storage-tree invariant** as a
whole, but it crosses out of the capture pipeline into the
production-kernel emission path. The user's "no tactical fixes / fix at
the source" rule could be read either way: the capture pipeline's
SOURCE bug is the leftover-walk reading aliased storage, which is X;
the kernel writer's SOURCE bug is the `pack[1]` aliasing itself, which
is Y.

I am not pre-committing because Y has codegen risk that I have not
confirmed bounded.

**Recommendation if the user wants me to pick one:** Option X, scoped
to the capture pipeline, with a clear in-line docstring explaining the
`pack[1]` aliasing precondition the rewrite is defending against.

### Ruled out

- **(Z, tactical) Dedupe inside `_make_node`** in `CMSValidator.py` —
  user has explicitly forbidden tactical patches at the validator
  layer.
- **(Z′, tactical) Dedupe inside `LoopBodyCaptureBuilder.append`** —
  same shape of fix; symptom-suppression rather than source-fix.
- **(Z″, tactical) Drop POST_LOOP captures whose `(slot_kind,
  mfma_index, id)` triple already exists in any earlier append** —
  same shape; symptom-suppression.

Each of these masks the kernel-writer aliasing without addressing it
and would re-surface immediately if the capture path picked up another
storage source that re-references the same leaves.

---

## §3 — Test surface (Q5)

The right pinning test is **mechanism-agnostic**: assert that no two
`TaggedInstruction`s in a single `LoopBodyCapture` reference the same
`rocisa_inst` Python object. This catches whatever the underlying cause
is (today: `pack[1]` aliasing; tomorrow: any new aliasing convention
that surfaces through any capture path).

Suggested test (drop into
`Tests/unit/test_capture_pipeline_checks.py`, addable to the validation
branch independently of 4up4):

```python
def test_no_double_capture_of_same_rocisa_inst_in_any_body(
    isa_infrastructure,
):
    """A single body's capture must not contain two TaggedInstructions
    that wrap the same Python rocisa instance.

    Pinned by xbi0: the leftover-pack walk in _loopBody's
    capture-default branch was emitting the same packCodeA leaf twice
    (once via pack[storeIdx*N], once via pack[1]) under
    ForceUnrollSubIter kernels, producing two TaggedInstructions with
    the same id(rocisa_inst) at different slot.sequence positions in
    the same body's POST_LOOP bucket.
    """
    # build canonical TF32 4x4 TN kernel ...
    cap = writer._last_default_capture
    for label, by_cp in (
        ("ML",   cap.main_loop),
        ("ML-1", cap.main_loop_prev),
        ("NGL",  cap.n_gl),
        ("NLL",  cap.n_ll),
    ):
        for cp, body in by_cp.items():
            seen = {}
            for ti in body.instructions:
                rid = id(ti.wrapped.rocisa_inst)
                if rid in seen:
                    raise AssertionError(
                        f"{label}[{cp}]: rocisa_inst id={rid} "
                        f"({type(ti.wrapped.rocisa_inst).__name__}) "
                        f"appears at slot {ti.slot} cat={ti.category} "
                        f"AND at slot {seen[rid].slot} "
                        f"cat={seen[rid].category}"
                    )
                seen[rid] = ti
```

Two notes:
1. The check should run at the level of the raw capture (before the
   data-flow graph builder filters categories), so that a future bug
   in a non-data-flow category (e.g., LRS / LWS) is also caught.
2. Use the canonical TF32 4×4 TN kernel as the production-shape pin
   (same kernel as the existing 18-test cluster). A SECOND simpler
   stub-based test that drives `LoopBodyCaptureBuilder.append`
   directly with a same-id pair and asserts the rule is also a useful
   layer of defense — but the kernel-shape test is the load-bearing
   one.

The test is independent of 4up4 because the invariant ("no double-
capture of the same rocisa Python object in a single body") was true on
the historical identity scheme as well — just unobserved because
`compare_graphs`'s `class_tag='PACK'` filter excluded pack-MFMAs from
the data-flow set.

---

## §4 — Test-suite impact (Q6)

For each of the 4 affected files in the brief, I traced one
representative test and classified.

### `test_ScheduleCapture.py::TestRealCmsKernelCapture::test_tf32_4x4_tn_capture_shape`

**What it asserts.** Calls `writer._getKernelSource(solution)` directly
(no try/except). Then reads `writer._last_cms_capture` and asserts shape
(num_codepaths, body keys, MFMA-per-body invariant).

**Why it's failing today.** `_getKernelSource` raises
`CaptureConsistencyError` from `compare_graphs` BEFORE returning, so
`writer._last_cms_capture` is unset when the test reads it. Once
double-capture is fixed, `compare_graphs` returns clean, the call
succeeds, and the assertions pass.

**Type:** (a) — fixed by the xbi0 fix landing.

### `test_cms_from_default.py::TestEndToEnd::test_emits_python_file_for_known_good_config`

**What it asserts.** Runs `default_schedule_to_cms` end-to-end on a
known-good kernel config, checks the emitted Python file's content and
the `ValidationReport`. Asserts `report.graph_diff == []` (i.e.,
`compare_graphs` returns empty).

**Why it's failing today.** Same root cause: the conversion pipeline
calls `_getKernelSource` which raises on the double-capture. After
fix, conversion succeeds and the report is clean.

**Type:** (a) — fixed by the xbi0 fix landing.

### `test_prologue_capture.py::test_whole_kernel_useplrpack_cms_matches_both_defaults`

**What it asserts.** Builds the kernel under both `UsePLRPack=True` and
`UsePLRPack=False`, then asserts `compare_graphs` and
`validate_edge_wait_coverage` both return empty residuals (after
filtering legitimate `TimingTooCloseFailure` entries).

**Why it's failing today.** `compare_graphs` raises before it can
return any failures. Fix: clear capture identity sets allow the
comparison to run and return empty (assuming the rest of the kernel
captures match — which they should, since the asymmetry was solely the
leftover-pack walk).

**Type:** (a) — fixed by the xbi0 fix landing.

### `test_cross_subiter_alu_carveout_real_kernel.py::test_real_kernel_validates_clean_with_carveout_engaged`

**What it asserts.** Builds the canonical TF32 4×4 TN kernel; calls
`compare_graphs(ref, subj)` and asserts `failures == []`.

**Why it's failing today.** Same: `compare_graphs` raises before it
returns. After fix: the cross-subiter ALU carve-out (which is what
this test validates the engagement of) absorbs the legitimate Pack3 ->
MFMA pipelining edges, so the failure list is empty.

**Type:** (a) — fixed by the xbi0 fix landing.

### Aggregate

**All 18 failing tests are type (a).** None of them assert the
HISTORICAL capture shape; they all assert end-to-end CMS-vs-default
validation cleanliness or build-pipeline survival. The xbi0 fix lands
them all green.

The 3 xfailed regression tests in
`test_dataflow_graph_emission_ordinal.py:376-471`
(`test_real_kernel_per_render_counts_match`,
`test_real_kernel_per_ordinal_logical_instruction_matches`,
`test_example_yaml_no_spurious_order_inverted_failures`) should also
flip green once the double-capture is fixed, per the xfail strings'
own statement.

---

## §5 — References

- `EMISSION_ORDINAL_DESIGN.md` §2.7-§2.8 (rocm-libraries-4up4) —
  documents the new identity tuple `(loop_index, canonical_render,
  emission_ordinal)` that surfaces the double-capture as a
  `CaptureConsistencyError`. The historical `class_tag='PACK'` slot
  excluded pack-MFMAs from the data-flow filter and absorbed the
  defect.
- `2LZD_INVESTIGATION.md` §1.B (rocm-libraries-2lzd) — documents
  PackA0/PackA3 cross-tagging in the per-iter capture path
  (`PackCodeAAllIters[u]` cross-tagged for the same canonical
  rendering). That residual is **conceptually separate from xbi0's
  POST_LOOP double-capture** but is the same family of "the kernel
  writer's pack-routing convention does not match what the capture
  pipeline assumes". Both should be on the user's radar; only the
  POST_LOOP one is what fires the `CaptureConsistencyError` in the 18
  failing tests.

---

## §6 — Notes for the reviewer

The brief framed the bug as "same id appears at PackA0 and PackA3" with
sequence numbers 4 and 44 in PRE_LOOP. The forensic capture I ran on
the same canonical kernel produced same id appearing at PackA3 and
PackA3 with sequence numbers 4 and 44 in **POST_LOOP**, not PRE_LOOP.
This is consistent with the slot.sequence numbers but inconsistent with
the slot_kind. I take that as evidence the brief consolidated two
overlapping symptoms (the per-iter PackA0/PackA3 cross-tagging from
2LZD §1.B AND the POST_LOOP double-append) into a single sentence; the
mechanism is the POST_LOOP double-append described above. The fix
recommendation is unaffected by this disambiguation because both
symptoms ultimately track back to the kernel writer's pack-routing
conventions, but the immediate fix surface is the POST_LOOP leftover
walk at `KernelWriter.py:4613-4628`.
