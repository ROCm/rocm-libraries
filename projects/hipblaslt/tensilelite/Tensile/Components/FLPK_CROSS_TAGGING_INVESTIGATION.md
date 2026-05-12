# FLPK — Default-side per-iter Pack cross-tagging investigation

Bead: `rocm-libraries-flpk`. Investigation only — no production code
changes. Forensic scripts (now removed) were used to confirm the
mechanism on the validation branch (`users/alvasile/validator_long_term_plans`)
without needing 4up4's identity scheme to surface the defect.

## TL;DR

- **Mechanism**: two distinct Python objects with identical canonical
  text get tagged with two different `PackA{u}` strings via separate
  capture sites in `_loopBody`'s default-capture path. Specifically: the
  PRE_LOOP slot for u=0 (subiter==0 pre-MFMA stream) tags items via
  `capture_id_to_cat` built from `PackCodeAAllIters[0]` -> `PackA0`; the
  POST_LOOP slot from the leftover-pack walk tags items via
  `leftover_id_to_cat` built from `PackCodeAAllIters[3]` -> `PackA3`.
  Each `localReadDo` call returns FRESH Python `Instruction` objects, so
  iter-0 and iter-3 produce different `id()` values for identical
  `v_cvt_pk_bf16_f32` text.
- **Independence verdict**: scenario **(a) Independent** from xbi0.
  Forensic id-equality test across all four bodies (ML-1, ML, NGL, NLL)
  shows **0 same-id cross-tag pairs, 80 diff-id cross-tag pairs**. xbi0's
  `pack[1]` aliasing fix (Option X or Y in `XBI0_CAPTURE_DEFECT_INVESTIGATION.md`)
  does NOT close flpk's diff-id pairs — they live in a different
  capture site and depend on distinct Python objects.
- **Brief framing partially incorrect**: the brief says
  "cross-tagging happens in all four bodies." Forensic shows it fires
  ONLY in ML-1 and ML (40 cross-tagged canons each) — NGL and NLL have
  ZERO cross-tagged canons. NGL/NLL still produce the
  `PackA0/PackA1/PackB0/PackB1` per-iter tag set the brief described,
  but those tags do NOT collide on canonical text within those bodies
  (the leftover-walk path is `_loopBody`-only; `_noLoadLoopBodyDefault`
  has no leftover-pack capture path).
- **Recommended fix candidates need user decision** between two
  principled options at the SOURCE (the kernel writer's pack-routing
  pipeline). Both are listed below in §3 with trade-offs.

---

## §1 — Mechanism (Q1, Q3, Q4)

### Q1 — Where the cross-tagging happens

`build_idmap` at `KernelWriter.py:4488-4502` (per-iter capture call) and
`:4588-4603` (leftover-pack capture call) are two SEPARATE invocations
of the same `ScheduleCapture.build_idmap` factory
(`ScheduleCapture.py:908-947`). Each invocation produces an idmap
keyed by category strings `PackA{u}` for `u in range(num_loop_iter)`,
with `PackCodeAAllIters[u]` as the source module. The map is then
inverted via `invert_idmap_to_id_to_category`
(`ScheduleCapture.py:950-975`), which raises `ValueError` if the same
id appears under two categories.

**The categorization itself is correct** — `invert_idmap_to_id_to_category`'s
duplicate-id check confirms that within a single invocation, every leaf
id is in at most one category. The cross-tagging arises because:

1. The per-iter capture call at `:4488-4502` runs ONCE per uIdx (inside
   the for-uIdx loop at `:4038`), each time with the CURRENT
   `len(LRCodeAAllIters)` as `num_loop_iter`. At u=0, `PackCodeAAllIters`
   has length 1 (only `PackCodeAAllIters[0]` is populated). At u=1 it
   has length 2, etc.
2. The leftover-pack capture call at `:4588-4603` runs ONCE at end of
   for-uIdx loop, with `num_loop_iter = len(LRCodeAAllIters) =
   kernel["LoopIters"]` (4 for TF32 4x4 TN).
3. `_makeSubIterSchedule` is called per-uIdx with the per-iter
   `capture_id_to_cat`. At u=0, `_captureSubIterToBuilder`
   (`KernelWriter.py:2601-2688`) walks the iterCode and tags items via
   id-lookup. Items emitted before the first MFMA go into PRE_LOOP slot
   (`:2669-2673`). At u=0, the per-iter `capture_id_to_cat` only
   contains tags for `PackCodeAAllIters[0]` (-> `PackA0`).
4. The leftover-pack walk at `:4613-4628` walks `pack[*]` and
   `packPre[*]` AFTER all iters complete, using
   `leftover_id_to_cat` (built from `PackCodeAAllIters[0..3]`). Leaves
   in `pack[1]` (aliased iter-3 packCodeA) get tagged via
   `leftover_id_to_cat[id(packCodeA_iter3)]` -> `PackA3`, written into
   POST_LOOP slot.

The two PRE_LOOP-PackA0 leaves and the POST_LOOP-PackA3 leaves are
distinct Python objects (different `localReadDo` invocations at iter 0
vs iter 3) that happen to render to identical
`v_cvt_pk_bf16_f32 v[X], v[Y], v[Z]` canonical text. This is a
**canonical-text collision across distinct Python objects produced by
distinct `localReadDo` calls**.

### Q3 — Cross-body sweep

Forensic scan across all four bodies on the canonical TF32 4x4 TN
kernel (`UseCustomMainLoopSchedule=1`):

| Body | pack-bucket counts | cross-tagged canons | same-id | diff-id |
|---|---|---|---|---|
| ML-1 | mfma/PackA0=34, mfma/PackB0=34, post_loop/PackA3=40, post_loop/PackB3=20, pre_loop/PackA0=6, pre_loop/PackB0=6 | 40 | 0 | 40 |
| ML   | mfma/PackA0=34, mfma/PackB0=34, post_loop/PackA3=40, post_loop/PackB3=20, pre_loop/PackA0=6, pre_loop/PackB0=6 | 40 | 0 | 40 |
| NGL  | mfma/PackA0=14, mfma/PackA1=20, mfma/PackB0=14, mfma/PackB1=20, pre_loop/PackA0=6, pre_loop/PackB0=6 | 0 | 0 | 0 |
| NLL  | mfma/PackA0=14, mfma/PackA1=20, mfma/PackB0=14, mfma/PackB1=20, pre_loop/PackA0=6, pre_loop/PackB0=6 | 0 | 0 | 0 |

**Brief framing correction (Q3 answer)**: NGL and NLL bodies do NOT
exhibit the cross-tagging. The cross-tagging is ML-only — confined to
the `_loopBody` capture path's POST_LOOP leftover walk
(`KernelWriter.py:4613-4628`).
`_noLoadLoopBodyDefault` (`:3623-3635`) uses a different per-iter
factory (`build_id_to_category_per_iter` at
`ScheduleCapture.py:1017-1108`) and has NO leftover-pack capture step,
so the canonical-text-collision conditions are never set up there.

Empirical aggregate: **80 diff-id cross-tag canons total, 0 same-id**.

### Q4 — CMS-side asymmetry

CMS-side dispatch (`Tensile/Components/CustomSchedule/dispatch.py:81-194`)
calls `build_idmap` ONCE at `:136-149` with the same
`PackCodeAAllIters` source. The CMS macro emits each idMap entry
EXACTLY ONCE per `optSchedule[key][cp][i]` reference (the `scheduleInst`
helper at `:207-230` consumes the instructionList by index). There is
no "leftover walk" or "pre-loop pre-MFMA capture" on the CMS side —
the macro is a single linear emission keyed by mfmaIndex. So a leaf
that has identical canonical text to another leaf in a different
`PackCodeAAllIters[u]` bucket is emitted at exactly one position with
exactly one tag.

The default-side has TWO distinct capture sites
(per-iter PRE_LOOP via `capture_id_to_cat`, leftover POST_LOOP via
`leftover_id_to_cat`). Both write into the same body's
`LoopBodyCaptureBuilder._instructions` list. They share the same
underlying `build_idmap` factory but operate on DIFFERENT slices of
`PackCodeAAllIters`:

- per-iter `capture_id_to_cat` at u=0 covers
  `PackCodeAAllIters[0]` only (length-1 list at that call site).
- leftover `leftover_id_to_cat` covers `PackCodeAAllIters[0..3]` (full
  list at end-of-for-uIdx).

Because each `localReadDo` call at iter `u` produces a fresh
`Instruction` object that goes into BOTH `pack[packStoreIdx*N]` AND
`PackCodeAAllIters[u]` (lines `:4205, 4208-4209`), the leftover walk
sees iter-3's packCodeA in `pack[1]` (xbi0's aliasing) and tags it
PackA3 via the full-coverage `leftover_id_to_cat`. The PRE_LOOP slot
at u=0 captures items whose `_captureSubIterToBuilder` lookup against
the partial `capture_id_to_cat` happened to return PackA0 — those items
are distinct Python objects from the iter-3 ones, with identical
rendered text but different `id()` values.

---

## §2 — Independence from xbi0 (Q2)

**Verdict: scenario (a) Independent.**

Forensic id-equality test across all four bodies of the canonical
TF32 4x4 TN kernel produced **0 same-id cross-tag pairs and 80 diff-id
cross-tag pairs**. xbi0's mechanism is the SAME `id(rocisa_inst)`
appended twice to one body's instruction list under the SAME category
tag (PackA3+PackA3, PackB3+PackB3). flpk's mechanism is TWO DIFFERENT
`id(rocisa_inst)` values appended at different slot_kinds with
different tags (PackA0@PRE_LOOP vs PackA3@POST_LOOP).

xbi0's Option X (restructure leftover walk to dedup against the
per-iter walk's already-appended id-set) and Option Y (eliminate
`pack[1].add(...)` aliasing in the kernel writer) **do NOT address
flpk**:

- **Option X removes same-id duplicates**. The flpk pairs already have
  different ids, so dedup by `id()` is a no-op for flpk.
- **Option Y removes `pack[1]` aliasing**. The leftover walk would no
  longer find iter-3's packCodeA in `pack[1]`. But it WOULD still find
  it in `pack[3]` (the actual `pack[packStoreIdx*N]` slot) — the
  aliasing hides the residue in TWO buffers, but removing the alias
  still leaves the residue in ONE buffer. The leftover walk tags
  whatever it finds via `leftover_id_to_cat` -> `PackA3`. The PRE_LOOP
  PackA0 tag for the canonically-identical (but distinct id) prefetch
  leaf is unchanged.

**Both fixes are necessary on their own; neither subsumes the other.**

There IS a shared upstream cause at the design level (the kernel
writer's pack-routing convention spreads the same logical pack
operation across multiple bookkeeping containers without an
"ownership" invariant). But the SURFACE mechanisms — same-id slot-list
double-append vs different-id canonical-text collision — are
mechanically distinct and require distinct fixes.

---

## §3 — Recommended fix approach (Q5)

Both candidates address the **canonical-text collision under the
default-side capture's two-call-site categorization**. Neither
introduces CMS-coupled categorization. Neither is a tactical
validator-layer dedup. There IS a meaningful architectural trade-off
and the user should weigh in.

### Option F1 — Unify the two default-side capture-site idmaps

**Where.** `KernelWriter.py:4471-4528` (per-iter capture builder
init + invocation) and `:4571-4630` (leftover-pack capture invocation).

**Invariant established.** "The default-side capture path uses ONE
unified `id_to_category` map per body, built once from the full
`PackCodeAAllIters[0..LoopIters-1]` range, and consulted by BOTH the
per-iter `_captureSubIterToBuilder` walks AND the leftover-pack walk.
Within a single body, no canonical text is tagged twice with different
category strings."

**Mechanism of the fix.** Build `leftover_id_to_cat`-equivalent
(full-range) ONCE at the start of the for-uIdx loop (or lazily on
first per-iter invocation), reuse for every per-iter invocation AND
the post-loop leftover. The PRE_LOOP entries that today get tagged
`PackA0` (because they live in `PackCodeAAllIters[0]`) would still be
tagged `PackA0`. The POST_LOOP entries that today get tagged `PackA3`
(because they're in `PackCodeAAllIters[3]`) would still be tagged
`PackA3`. The cross-tagging would PERSIST under this rule because the
two leaves have different `PackCodeAAllIters[u]` source positions.

**Conclusion: F1 alone is insufficient.** Unifying the idmap doesn't
eliminate the divergence — it just confirms that the default-side has
consciously categorized two different objects with two different tags
even though they render identically. F1 is necessary as a precondition
for some downstream invariants (e.g. removing the assert-raising
behavior of `invert_idmap_to_id_to_category` — see §6 follow-ups in
2LZD), but it doesn't fix flpk on its own.

### Option F2 — Eliminate the PRE_LOOP capture of pack-leaves whose canonical text matches a leftover entry

**Where.** `KernelWriter.py:4509-4514` (the prefetch_pack tag
setdefault) and `:2643-2688` (`_captureSubIterToBuilder` walk).

**Invariant established.** "Pack leaves are captured at most once per
body (across PRE_LOOP and POST_LOOP slots combined), at the
chronologically-earlier slot, with the corresponding `PackA{u}` tag."

**Mechanism of the fix.** When the leftover-walk visits `pack[*]` at
end-of-uIdx-loop, mark by canonical_str (not by id) the set of pack
canonical texts already emitted into the body's PRE_LOOP slot in
subiter==0. For canonical texts that ARE already in PRE_LOOP, skip the
POST_LOOP append. For canonical texts that AREN'T, append normally as
PackA{u} POST_LOOP.

**Trade-offs.**
- Pro: removes the visible canonical-text collision. Both the count
  delta in `2LZD §1.B` table (`PackA0+20, PackA3+20, PackB0+20`) and
  the would-be `compare_graphs` failures collapse.
- Pro: preserves the per-iter PRE_LOOP categorization (PackA0, PackB0
  etc.) which is semantically correct (the prefetch-source pack chain
  belongs to iter 0's lookahead).
- Con: introduces a "dedup by canonical_str" step at one capture site.
  This is principled (the invariant is "the same canonical text is the
  same instruction modulo register numbering") but it COULD mask a real
  divergence where the kernel writer legitimately needs two separate
  emissions of the same canonical text. None of the production kernels
  exercise that legitimately — but it's a contract worth documenting.
- Con: requires the leftover-walk to know about the per-iter PRE_LOOP
  walks' canonical_str sets, introducing a small internal-state read
  on `self._capture_context.builder._instructions`.

### Option F3 — Eliminate the leftover-pack walk's POST_LOOP capture entirely

**Where.** `KernelWriter.py:4613-4628`.

**Invariant established.** "Pack leaves that didn't enter any iter's
`_makeSubIterSchedule` invocation are NOT captured." The leftover walk
becomes a no-op (or a strict-mode assertion that the residue is empty,
to catch legit bugs).

**Why this is plausible.** xbi0 and 2LZD both observe that the
leftover-pack walk's POST_LOOP entries are an artifact of the kernel
writer's circular-buffer pack-routing under `ForceUnrollSubIter`. The
real CMS emission consumes ALL pack content via `optSchedule[PackA{u}]`
in the macro; there's no "residue" in CMS. The leftover walk on the
default-side is a fictitious bookkeeping that exists to make the count
match CMS, but `2LZD §1` already documented that the contract this
serves is itself rejected (the user has rejected shadow-vs-CMS
comparison entirely; see `2LZD_INVESTIGATION.md §6`).

**Trade-offs.**
- Pro: removes the entire POST_LOOP leftover-pack capture surface.
  Eliminates flpk AND xbi0 in one shot (xbi0's slot-list double-append
  also lives in this code path).
- Pro: aligns with the `2LZD §6` decision to retire shadow-vs-CMS
  comparison. The leftover-pack is a feature of the shadow contract;
  if the shadow's contract is being retired (Approach A is the chosen
  path), the leftover-pack capture is being deleted as part of that
  larger work.
- Con: changes default-side body shapes today (the 80 + 60 nodes
  per body that 2LZD attributes to leftover-pack disappear). Any test
  that pins those shapes would need updates. But per `2LZD §6` those
  tests are themselves on the chopping block.
- Con: largest blast radius of the three. F3 is properly part of the
  Approach-A migration in 2lzd's umbrella, not a standalone fix.

### Architectural trade-off

- F1 alone: insufficient (doesn't fix flpk).
- F2 alone: principled and minimal; addresses flpk and is independent
  of xbi0 / 2lzd timeline.
- F3 alone: principled and maximal; addresses flpk + xbi0 + 2lzd
  cross-link in one shot. Requires the meta-bead's Approach-A path to
  be the active direction (per `2LZD §6.2` and `rocm-libraries-71hw`
  meta-bead, this IS the chosen direction).

**The recommendation requires a user decision between F2 (scoped to
flpk only) and F3 (folded into the Approach-A migration of 2lzd).**

If F3 is chosen, flpk is closed by the same code-deletion that closes
the leftover-pack-walk surface for xbi0 and 2lzd. If F2 is chosen,
flpk is closed independently and earlier (xbi0 + 2lzd remain on their
own timelines).

**My recommendation if pressed**: **F3, executed as part of the
Approach-A migration in `rocm-libraries-71hw`.** F2 is correct but its
work would be discarded by F3's later landing. F3 is also more
architecturally honest (the leftover-pack capture is
shadow-contract-specific bookkeeping; deleting the shadow contract
deletes its bookkeeping too).

### Ruled out

- **Adding category-collapse logic in `compare_graphs`** — explicitly
  forbidden by the brief's Discipline rule 3.
- **Special-casing the cross-tagging in the validator to dedup** —
  explicitly forbidden by Discipline rule 2.
- **Modifying the rocisa-derived data-flow filter to merge PackA0 and
  PackA3** — same reason as above; CMS-shaped category strings do not
  belong inside identity / role / edge logic.

---

## §4 — Test surface (Q6)

The right pinning test is **mechanism-agnostic**: assert that no two
`TaggedInstruction`s in a single body share the same canonical_str
under different category tags.

This is a STRICTER invariant than xbi0's "no two TaggedInstructions
share the same `id(rocisa_inst)`" — flpk's pairs have different ids
but identical canonical_str. xbi0's invariant catches xbi0; the
extended invariant below catches BOTH xbi0 and flpk.

```python
def test_no_canonical_text_cross_tagged_in_any_body(isa_infrastructure):
    """Sibling to xbi0's same-id test: no two TaggedInstructions in a
    single body share the same canonical_str under different category
    tags.

    Pinned by flpk: the default-side _loopBody capture has two distinct
    sites (per-iter PRE_LOOP via capture_id_to_cat from
    PackCodeAAllIters[0..u], post-loop POST_LOOP via leftover_id_to_cat
    from PackCodeAAllIters[0..LoopIters-1]) that can tag two different
    Python objects with identical canonical text under different
    PackA{u} category strings.
    """
    # build canonical TF32 4x4 TN kernel
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
                # Restrict to pack categories — other categories
                # legitimately may have repeated canonical text
                # (e.g. SYNC s_waitcnt(0) repeated).
                if not (ti.category.startswith("PackA") or
                        ti.category.startswith("PackB")):
                    continue
                canon = ti.wrapped.canonical_str(ti.wrapped.rocisa_inst)
                key = (canon, ti.slot.slot_kind)
                # Allow same canonical_str within ONE (slot_kind, category)
                # bucket. Forbid same canonical_str across different
                # category strings within the same body.
                seen.setdefault(canon, ti.category)
                if seen[canon] != ti.category:
                    raise AssertionError(
                        f"{label}[{cp}]: canonical text {canon!r} "
                        f"appears under cat={ti.category} and earlier "
                        f"under cat={seen[canon]} (slot_kind={ti.slot.slot_kind}, "
                        f"id={id(ti.wrapped.rocisa_inst)})"
                    )
```

Three notes:
1. Run at the level of the raw capture (before the data-flow graph
   builder filters categories) so that future bugs in non-data-flow
   categories are also caught.
2. Use the canonical TF32 4x4 TN kernel (the same kernel as the 18-test
   cluster) as the production-shape pin.
3. The test is independent of 4up4 because the invariant ("no
   canonical-text cross-tagging in a single body") was true on the
   historical identity scheme as well — just unobserved because
   `compare_graphs`'s `class_tag='PACK'` filter excluded pack-MFMAs
   from the data-flow set.

---

## §5 — Test-suite impact (Q7)

### Currently-passing tests on validator_long_term_plans

**No currently-passing test pins the existing PackA0/PackA3 cross-tag
split on the validation branch.** The cross-tagging is latent here
(2lzd noted: "the dataflow graph collapses them via `nodes_by_identity`
(same canonical_str -> same node), so the double-tagging in (B)/(B-2)
above doesn't always surface as two nodes — it just inflates the
LoopBodyCapture's instruction count and downstream node-count, which
then matches between ref and subj because both sides inflate
identically").

The `_dump_carveout_assembly.py` script and `default.s` fixture
file CONTAIN the PackA0+PackA3 cross-tag evidence visibly, but those
are diagnostic dumps, not assertion targets. No pytest assertion
depends on `PackA0@PRE_LOOP` and `PackA3@POST_LOOP` being separate
node identities.

Some tests reference `PackA0`/`PackA3` strings as expected values
(grep hits in `test_quad_cycle_dispatch_table.py`,
`test_pre_post_loop_label_disambiguation.py`,
`test_dataflow_graph_phantom_edges.py`, `test_validate_pack_graph.py`,
etc.) but these are all CMS-side tests that pin CMS-side category
strings (which are not in scope for flpk's fix). None pin the
DEFAULT-side cross-tag split.

**Verdict: F2 or F3 should not break any currently-passing test on
the validation branch.**

### Currently-failing tests on 4up4

xbi0's investigator stated 18 unit tests are failing on 4up4 due to
the POST_LOOP slot-list double-append. Per xbi0's §4, all 18 are
"type (a)" — the failure is `_getKernelSource` raising
`CaptureConsistencyError` from `compare_graphs`, before the test's
own assertions can run.

**Will any currently-failing 4up4 test go green for flpk reasons?**
Probably not — but worth verifying. The 4up4 identity scheme
(`(loop_index, canonical_render, emission_ordinal)` per
`EMISSION_ORDINAL_DESIGN.md §2.7-§2.8`) keys on canonical_render. Two
nodes with the same canonical text in the same body would either
collapse into one identity (if the emission_ordinal also matches) or
diverge (if it doesn't). flpk's pairs have different `id(rocisa_inst)`
but identical canonical text and live at different stream positions,
so they'd have different emission_ordinals and produce two distinct
identity tuples.

The result depends on which path raises first inside `compare_graphs`.
xbi0's mechanism likely raises first (same id appended twice produces
a `CaptureConsistencyError` very early in graph build, before
canonical-text-based identity collapse kicks in). After xbi0 is
fixed, flpk's diff-id pairs would EACH become a separate
`GraphNode` (because the identity tuple differentiates them via
emission_ordinal) and `compare_graphs` would then see N PackA0+PackA3
nodes on the default side vs ~N/2 PackA{u} nodes on the CMS side
(since CMS only emits each PackCodeAAllIters[u] entry once). That's
a count divergence that would surface as a NEW failure class once
xbi0 is fixed.

**Verdict for Q7**: fixing xbi0 alone would likely UNCOVER additional
failures attributable to flpk on 4up4. The two beads need to be
landed together (or at minimum, flpk lands FIRST or AT-LEAST-WITH
xbi0). Per the meta-bead `rocm-libraries-71hw` framing in 2LZD §6.2,
the entire shadow-capture-comparison surface is being retired under
Approach A; F3 (delete the leftover-pack walk) closes flpk + xbi0 +
2lzd's leftover-pack mechanism in one shot and is consistent with
that direction.

**Verdict for the validation branch**: F2 or F3 closes flpk
independently. No currently-passing test breaks. No currently-failing
test exists.

---

## §6 — References

- `2LZD_INVESTIGATION.md §1.B` — original PackA0/PackA3 cross-tagging
  evidence from `default.s:4-11`. Found in
  `Tensile/Components/2LZD_INVESTIGATION.md`.
- `2LZD_INVESTIGATION.md §6` and §6.2 — user decision rejecting
  shadow-vs-CMS comparison; meta-bead `rocm-libraries-71hw`
  consolidation under Approach A.
- `XBI0_CAPTURE_DEFECT_INVESTIGATION.md` — `pack[1]` aliasing
  same-id slot-list double-append mechanism. Found in
  `Tensile/Components/XBI0_CAPTURE_DEFECT_INVESTIGATION.md` (4up4
  worktree only).
- `EMISSION_ORDINAL_DESIGN.md §2.7-§2.8` — 4up4's identity tuple
  `(loop_index, canonical_render, emission_ordinal)` that surfaces
  capture-side defects as `CaptureConsistencyError`s. Found in
  `Tensile/Components/EMISSION_ORDINAL_DESIGN.md`.
- `KernelWriter.py:4488-4502` — per-iter capture `build_idmap` call
  (sets up `capture_id_to_cat` from `PackCodeAAllIters[0..u]`).
- `KernelWriter.py:4509-4514` — prefetch-pack tag setdefault walk
  (PackA{plrIdx} for prefetch leaves; empty under `usePLRPack=True`).
- `KernelWriter.py:4588-4603` — leftover-pack capture `build_idmap`
  call (full-range `PackCodeAAllIters[0..LoopIters-1]`).
- `KernelWriter.py:4613-4628` — leftover-pack walk that emits
  POST_LOOP-tagged TaggedInstructions; this is the surface site for
  flpk's PackA3 tag and also xbi0's same-id double-append.
- `KernelWriter.py:4205, 4208-4209, 4211` — kernel writer's pack
  routing under `ForceUnrollSubIter`: same `packCodeA` added to
  `pack[packStoreIdx*N]`, to `PackCodeAAllIters[uIdx]` (for CMS), and
  aliased into `pack[1]`.
- `KernelWriter.py:3623-3635` — `_noLoadLoopBodyDefault`'s per-iter
  `build_id_to_category_per_iter` call (different factory; no
  leftover-pack walk; no cross-tagging surface in NGL/NLL).
- `Tensile/Components/CustomSchedule/dispatch.py:81-194` — CMS-side
  `customMainLoopSchedule`; `:136-149` is the single `build_idmap` call
  CMS uses; `:200-234` is the macro emission that consumes each
  `idMap[key]` exactly once.
- `Tensile/Components/ScheduleCapture.py:908-947` — `build_idmap`
  factory (single source of truth for category schema).
- `Tensile/Components/ScheduleCapture.py:950-975` —
  `invert_idmap_to_id_to_category` (raises ValueError on duplicate id;
  this is why each call site's idmap is internally consistent).
- `Tensile/Components/ScheduleCapture.py:1017-1108` —
  `build_id_to_category_per_iter` (sibling factory used by
  `_noLoadLoopBodyDefault`).
