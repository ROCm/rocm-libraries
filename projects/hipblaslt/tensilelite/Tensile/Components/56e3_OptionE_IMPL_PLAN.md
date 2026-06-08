# rocm-libraries-56e3 Option E Implementation Plan

**Status:** planning  
**Bead:** `rocm-libraries-56e3` — producer-discrimination loss in byte-key edge matching  
**Prereq commit:** C3d (`5b187061007c`) already landed  
**Goal:** resurface 5 detection tests broken by C3d while preserving the n7og fix  

---

## §1 Scope

Option E augments `DataflowGraph.edge_keys()` with the producer node's source identity —
the iter-blind, register-rename-blind subset of the existing 3-tuple identity — so that
two distinct producers writing the same byte footprint to the same resource produce
distinct edge keys and do not cancel in set-diff. The parallel change to
`compare_graphs`'s `ref_edges_by_key` construction mirrors the new tuple shape exactly.
No new compare_graphs phase is introduced; the existing `diagnose_missing_edge` phased
classifier fires as before on the residual missing keys, emitting the same Failure types
the currently-failing tests pin. One additional step is required for synthetic test
fixtures: they must populate `source_module_id` on their `TaggedInstruction` objects so
the new discriminator field can distinguish producers even when `canonical_render` is
excluded from the key.

---

## §2 Investigation Findings

### A. Current `edge_keys()` state post-C3d

`CMSValidator.py:1312–1315`. The 6-field tuple is exactly:

```python
(e.producer_write_byte_key, e.consumer_read_byte_key,
 e.edge_kind, e.intra_operand_byte_offset,
 e.src_operand_slot, e.sink_operand_slot)
```

The `compare_graphs` `ref_edges_by_key` construction at `CMSValidator.py:1780–1785`
mirrors this shape one-for-one, same six fields, same order.

**Option E placement:** the two new producer-identity fields go at the **front** —
before `producer_write_byte_key`. They are producer attributes, not byte-content
attributes. Leading with producer identity makes the tuple self-describing: the first
element(s) describe the source, then what it wrote, then what the consumer read. The
new 8-field tuple is:

```
(producer_source_module_id, producer_emission_ordinal,
 producer_write_byte_key, consumer_read_byte_key,
 edge_kind, intra_operand_byte_offset,
 src_operand_slot, sink_operand_slot)
```

### B. DataflowEdge field availability

`GraphNode` (`CMSValidator.py:898–927`) carries:
- `identity: tuple` — the 3-tuple `(canonical_render, source_module_id, emission_ordinal)`,
  assigned at `_make_node` (`CMSValidator.py:1783`) via `tagged_inst.identity_for(body_label)`.
- `tagged_inst: "TaggedInstruction"` — back-reference.

`TaggedInstruction` (`ScheduleCapture.py:414–551`) carries:
- `source_module_id: Optional[str] = None` (`ScheduleCapture.py:460`)
- `emission_ordinal: int = -1` (`ScheduleCapture.py:449`)

Access paths (both equivalent):
- Via identity 3-tuple: `node.identity[1]` = `source_module_id`, `node.identity[2]` = `emission_ordinal`
- Via tagged_inst: `node.tagged_inst.source_module_id`, `node.tagged_inst.emission_ordinal`

The identity-tuple slicing approach (`node.identity[1]`, `node.identity[2]`) is preferred
because it avoids introducing a new property and makes the derivation from the identity
contract explicit.

**Population:** `_make_node` populates `identity` for every production GraphNode. Synthetic
GraphNode objects created directly (test stubs) carry whatever identity tuple was assigned.
The `make_capture` fixture helper (`dataflow_fixtures.py:339–370`) calls
`assign_emission_ordinals` (`ScheduleCapture.py:783–823`), which correctly populates
`emission_ordinal` on each `TaggedInstruction`. `source_module_id` is NOT set by any
fixture builder — it remains `None` unless the caller explicitly passes it.

### C. None-source_module_id handling — critical finding

**This is the key design constraint surfaced during investigation.**

`assign_emission_ordinals` (`ScheduleCapture.py:783–823`) keys its counter on
`(canonical_render, source_module_id)`. When `source_module_id=None`, the counter key
becomes `(canonical_render, None)`. Two instructions with **different** canonical renders
AND `source_module_id=None` both receive `emission_ordinal=0` from independent counter
buckets.

Concrete case (SCC test, `test_gr_simple` conflict 1):
- Default capture producer A = `SCmpEQU32(s50, s51)`: render key `("s_cmp_eq_u32 ...", None)`, ordinal 0. `source_identity = (None, 0)`.
- Subject capture clobber B = `SAddU32(s80, s80, s81)`: render key `("s_add_u32 ...", None)`, ordinal 0. `source_identity = (None, 0)`.

Both have `source_identity = (None, 0)`. Both write `producer_write_byte_key = (('scc', 0),)`.
The 8-field tuple would be **identical** for the REF edge `A→C` and the SUBJ edge `B→C` →
cancellation persists → no Failure emitted → test still fails.

**The architecture memo's claim ("discriminates producers in all 5 failing tests") is
incorrect for synthetic fixtures with `source_module_id=None`.** The 2-tuple
`(source_module_id, emission_ordinal)` only discriminates when: (1) `source_module_id`
values differ, OR (2) `source_module_id` is the same AND both producers have the same
canonical render (same counter bucket) with different ordinals (different positions in
the sorted stream). Neither condition holds for the SCC test fixtures.

**Resolution:** Synthetic test fixtures must populate `source_module_id` on their
`TaggedInstruction` objects with distinct string labels that model the source-module
context of each instruction. The fixture builders (`dataflow_fixtures.py`) and/or the
test-local `_tag()` helpers must accept a `source_module_id` kwarg and thread it through.
This is a principled fix, not a workaround: in production, every instruction from a named
generator Module has a non-None `source_module_id`. The synthetic `source_module_id=None`
default is a modeling shortcut that does not survive the new discriminator.

**None-handling in the key tuple itself:** `None` is hashable in Python. The key
`(None, 0, ...)` is a valid set element. There is no need to reject or special-case
`source_module_id=None` during key construction. The failing-test fix is upstream (fixture
builders), not inside `edge_keys()`.

**Empirical probe validation:** The empirical probe (`56e3_OPTION_E_EMPIRICAL_PROBE.md`)
found that all 20 `mult_differs` groups had `source_module_id=None` — these are scheduling
primitives (s_waitcnt, s_nop, s_barrier) that do not form `raw_intrawave` /
`lds_raw_intrawave` edge endpoints (they carry no reads/writes). They do not enter the
`edge_keys()` tuple as producer or consumer. This confirms that None-source
instructions are not relevant to the producer-discrimination problem in production
captures — but they ARE relevant in synthetic test fixtures.

### D. compare_graphs `ref_edges_by_key` site

`CMSValidator.py:1780–1785`:

```python
ref_edges_by_key = {}
for e in reference.edges:
    key = (e.producer_write_byte_key, e.consumer_read_byte_key,
           e.edge_kind, e.intra_operand_byte_offset,
           e.src_operand_slot, e.sink_operand_slot)
    ref_edges_by_key.setdefault(key, e)
```

This must be updated to the new 8-field shape in exact parallel with `edge_keys()`. The
`e.producer.identity[1]` and `e.producer.identity[2]` expressions on the `DataflowEdge`'s
`producer` node give `source_module_id` and `emission_ordinal` respectively.

### E. Other callers of `edge_keys()`

Non-test callers:
- `compare_graphs` (`CMSValidator.py:3756–3757`): production caller; migrated here.
- `repro_cross_subiter_artifact.py:450–451`: standalone repro script (not in test suite,
  not automatically run). Contains its own `ref_edges_by_key` construction at lines 466–
  468. Both uses must be updated for consistency, though the repro script is not a
  correctness gate.

Test callers:
- `test_dataflow_graph_comparison.py:1017–1018, 1044`: tests equality/inequality of
  `edge_keys()` sets. Both assertions use `==` / `!=` on the set; they do not destructure
  individual tuple elements. Under the new 8-field shape, equality semantics are
  unchanged — the assertions still hold.
- `test_dataflow_graph_register_gaps.py:3146`: asserts `g_a.edge_keys() != g_b.edge_keys()`.
  Same reasoning; no destructuring.
- `test_n7og_edge_keys_multifixture.py:277` (via `_edge_keys_for_capture`): computes
  symmetric difference of edge_keys sets between SHADOW and CMS captures. The n7og
  assertion is that this difference is 0. Under Option E, adding `source_module_id +
  emission_ordinal` at the front: SHADOW and CMS producers for the same physical
  instruction share `source_module_id` and `emission_ordinal` (same source module, same
  counter), so the 8-tuple is still byte-equal across sides. The 0-mismatch result is
  preserved. See §F for the full n7og reasoning.

No other callers found. `edge_keys()` is not exported from `CMSValidator.py`'s public
API surface.

### F. Test expectations — failure delta prediction

**Tests that should RESURFACE (become PASSING after Option E):**

These tests currently fail because `compare_graphs` returns an empty failure list where
it should return specific Failures. After Option E, the REF edges are no longer
cancelled by SUBJ edges with the same byte-key but different source identity → the
missing-key residual is non-empty → `diagnose_missing_edge` fires → correct Failure
emitted → test assertion passes.

BUT: resurfacing requires the fixture `source_module_id` fix (§C) to be in place first.
Without that fix, `source_module_id=None` on both producers means the 2-tuple
`(None, ordinal)` doesn't discriminate — the tests remain broken under Option E alone.

- `TestValidateSCCOverlap` × 6 methods (12 conflict assertions across `test_gr_simple`,
  `test_gr_declaration_order`, `test_gr_interval`, `test_gr_noshadow`, `test_lws`,
  `test_gr_inc_together`): emit `OverriddenInputFailure`. Require fixture source_module_id
  fix. (All 6 methods fail post-C3d; all 6 use the same `_build_scc_clobber_pair`
  helper with `source_module_id=None` — same collision pattern applies.)
- `test_validate_pack_graph.py::TestSwapPackGraph::test_pack_before_swap_orderinverted`:
  emits `OrderInvertedFailure`. Requires fixture source_module_id fix.
- `test_validate_gr_not_too_early_graph.py::TestGRNotTooEarlyDtlPlusLdsBufGraph::test_negative_one_prev_iter_lr0_not_drained`:
  emits `MissingWaitFailure`. Requires fixture source_module_id fix.

**Tests that must STAY passing (n7og non-regression):**

`test_n7og_edge_keys_multifixture.py::test_shadow_vs_cms_edge_keys_match` (all 3
fixtures: bpg11-tf32-4x4-tn, oplb-tf32-6x8-tn, bf16-256x256x64-tn).

Logical reason: The n7og worry is that UsePLRPack causes the SAME physical PackB3
instruction to render with different register names (T0_I0 vs X0_I0) under SHADOW vs CMS.
Under Option E, the producer key is `(source_module_id, emission_ordinal)` — NOT
`canonical_render`. The same physical instruction in SHADOW and CMS has the same
`source_module_id` (same generator module) and the same `emission_ordinal` (same counter
bucket within that module). The T/X register-name variation lives in `canonical_render`,
which is excluded from the key. Therefore the 8-tuple is byte-equal across SHADOW and CMS
for the same pipelined pack edge → set-diff cancels → 0 mismatches → test passes.

The empirical probe (`56e3_OPTION_E_EMPIRICAL_PROBE.md`) confirms: 0 `rw_differ` groups
across all fixtures and bodies. Every same-(canonical_render, source_module_id) multi-
emission group is `rw_match` — semantically interchangeable. The ordinal-flip scenario
(CMS reorders twin_α and twin_β) is benign because the twins have identical reads/writes.

**Expected failure count delta:**

End-of-C3d: 27 FAILED + 2 ERROR.
After Option E + fixture source_module_id fix: drop by 8 (6 SCC test methods + 1
pack-before-swap + 1 lr-not-drained). Predicted new total: **19 FAILED + 2 ERROR**.

(The bead description says "5 detection tests" because it counts test CLASSES not
individual methods; the actual `FAILED` line count in pytest output for the SCC class
is 6 — one per method. The safe claim is: all 56e3-regressions resurface as PASSING.)

### G. Stale docstring on `edge_keys()`

`CMSValidator.py:1248–1315`. The current docstring (C3d-updated) documents a 6-field
tuple and says "No identity references remain in the keying tuple." Both statements
become stale under Option E. The docstring needs:
1. Updated tuple shape (8-field, with leading producer source identity).
2. New section explaining WHY `source_module_id + emission_ordinal` are included
   (producer discrimination) and WHY `canonical_render` is excluded (register-rename-blind;
   T/X UsePLRPack variation lives in canonical_render, not in structural identity).
3. Updated comment in `compare_graphs` at `CMSValidator.py:3760–3770` to reference
   the 8-field shape.
4. Deletion of the known-limitation paragraph in `compare_graphs` docstring
   (`CMSValidator.py:3645–3651`) — it documents a limitation that Option E closes.
5. Update to the block-comment history at `CMSValidator.py:3709–3713`.

### H. Probe script as permanent regression guard

`Tensile/Tests/unit/test_56e3_option_e_probe.py` was authored as an empirical safety
probe. It should be **kept permanently as a regression guard** under the test suite.

Rationale: the probe's `groups_rw_differ=0` result is what makes Option E safe. If a
future CMS schedule introduces a case where the same `(canonical_render, source_module_id)`
group is emitted with different reads/writes on SHADOW vs CMS (ordinal-flip-with-distinct-
operations), the probe would emit a non-zero `rw_differ` count and alert. Without the
probe, that safety property becomes unverifiable.

Recommended rename: `test_56e3_option_e_probe.py` → `test_56e3_emission_ordinal_invariance.py`
(names the tested invariant directly; "probe" implies one-shot, "regression" is vague about scope).

### I. Expected failure delta

Per §F: end-of-C3d 27 FAILED + 2 ERROR → after Option E + fixture fix → 20 FAILED +
2 ERROR. The 7 tests removed from FAILED are exactly the 56e3-affected regressions. No
other tests should flip behavior.

---

## §3 Design — new `edge_keys()` tuple shape

```python
return {(e.producer.identity[1], e.producer.identity[2],
         e.producer_write_byte_key, e.consumer_read_byte_key,
         e.edge_kind, e.intra_operand_byte_offset,
         e.src_operand_slot, e.sink_operand_slot)
        for e in self.edges}
```

Where:
- `e.producer.identity[1]` = `source_module_id` (the generator module name; `None` for
  synthetic captures — see §C for why fixtures must populate this).
- `e.producer.identity[2]` = `emission_ordinal` (the per-`(canonical_render,
  source_module_id)` slot counter).
- The remaining 6 fields are unchanged from C3d.

**Placement justification:** producer identity fields lead the tuple. The first two fields
describe the source instruction (who wrote), the next two describe the bytes transferred
(what was written / what was read), the remaining four describe the dataflow topology
(how the edge is categorized). This ordering groups related concepts and matches the
semantic priority of the tuple: producer identity is the new discriminator that breaks
the collisions, and it should be visually prominent at the front.

**Properties preserved:**
- Hashable: `(str | None, int)` prefix; full tuple is set-usable.
- Iter-blind: iter copies of the same source instruction share `source_module_id` and
  `emission_ordinal` (they wrap the same underlying `TaggedInstruction`'s source module
  and counter slot).
- Register-rename-blind: `canonical_render` is absent; T/X naming variation does not
  enter the key.
- Producer-discriminating: two distinct source instructions have distinct
  `(source_module_id, emission_ordinal)` pairs when `source_module_id` is non-None.
  When `source_module_id=None`, two distinct-render instructions may share
  `(None, 0)` — hence the fixture source_module_id requirement (§C).

---

## §4 compare_graphs `ref_edges_by_key` migration

`CMSValidator.py:1780–1785` becomes:

```python
ref_edges_by_key = {}
for e in reference.edges:
    key = (e.producer.identity[1], e.producer.identity[2],
           e.producer_write_byte_key, e.consumer_read_byte_key,
           e.edge_kind, e.intra_operand_byte_offset,
           e.src_operand_slot, e.sink_operand_slot)
    ref_edges_by_key.setdefault(key, e)
```

Exactly parallel to the `edge_keys()` change. The comment block above this construction
(`CMSValidator.py:3760–3770`) is updated to describe the 8-field shape. The
`setdefault` semantics are unchanged: when multiple ref-edges share the same 8-tuple
key (cross-body pipelining of the same physical bytes from the same producer), the
first is taken as representative. Under Option E, the producer source identity fields
further reduce the ambiguity of "same key" — two edges sharing a key now also share
the same producer source module and ordinal slot, making "representative" more precise.

---

## §5 Step-by-step implementation order

1. **Fix fixture `source_module_id` population** (prerequisite for test resurfacing).

   - `Tensile/Tests/unit/dataflow_fixtures.py`: add `source_module_id: Optional[str] = None`
     to each `TaggedInstruction` constructor call in the factory functions (`make_lr`,
     `make_lw`, `make_gr`, `make_mfma`, `make_swaitcnt`, `make_sbarrier`, `make_snop`,
     any SCC builders). Thread an optional `source_module_id` kwarg through each factory.
   - `Tensile/Tests/unit/test_ValidateSCCoverlap.py`: update `_tag()`, `_producer_factory`,
     `_clobber_factory`, `_consumer_factory` to assign distinct `source_module_id` strings
     per instruction role (e.g. `"scc_producer"`, `"scc_clobber"`, `"scc_consumer"`).
   - `Tensile/Tests/unit/test_validate_pack_graph.py`: update fixture construction in
     `TestSwapPackGraph::test_pack_before_swap_orderinverted` to assign distinct
     `source_module_id` strings per instruction.
   - `Tensile/Tests/unit/test_validate_gr_not_too_early_graph.py`: same for the LR0/LR1/GR
     nodes in `test_negative_one_prev_iter_lr0_not_drained`.
   - No change to `assign_emission_ordinals` logic — it already handles
     `(canonical_render, source_module_id)` keys correctly.

2. **Update `DataflowGraph.edge_keys()`** (`CMSValidator.py:1312–1315`): prepend
   `e.producer.identity[1]` and `e.producer.identity[2]` to the tuple as specified in §3.

3. **Update `compare_graphs` `ref_edges_by_key` construction** (`CMSValidator.py:1780–1785`):
   parallel 8-field key as specified in §4.

4. **Update `edge_keys()` docstring** (`CMSValidator.py:1248–1311`): reflect 8-field
   shape; add "Producer discrimination" section explaining `source_module_id +
   emission_ordinal` inclusion and `canonical_render` exclusion; update the
   "Hashability" and "No identity references" stanzas.

   Additionally: fix the stale `GraphNode.identity` field comment at `CMSValidator.py:918`
   — it currently says `(canonical_render, emission_ordinal)` but `identity_for()` now
   returns a 3-tuple `(canonical_render, source_module_id, emission_ordinal)`. Update the
   comment to match. (This is a pre-existing stale comment, not introduced by Option E,
   but cleanup belongs here since Option E is the first code to index `identity[1]` and
   `identity[2]` explicitly.)

5. **Update `compare_graphs` docstring and history comment** (`CMSValidator.py:3636–3714`):
   - Update the opening summary paragraph to reflect 8-field key.
   - Delete the "Known limitation (rocm-libraries-56e3)" paragraph (`CMSValidator.py:3645–3651`).
   - Delete the "Residual limitation (rocm-libraries-56e3)" comment (`CMSValidator.py:3709–3713`).
   - Add a "56e3 resolution" entry to the history block.

6. **Update `repro_cross_subiter_artifact.py`** (`repro_cross_subiter_artifact.py:450–451`):
   The `edge_keys()` calls at lines 450–451 return sets of tuples; the set-diff at line
   452 (`missing_keys = ref_keys - subj_keys`) is agnostic to tuple shape, so the diff
   semantics are preserved under the new 8-field shape without code change. HOWEVER: the
   `ref_edges_by_key` construction at lines 466–468 uses a SEPARATE diagnostic key schema
   (`_role + position + slot + edge_kind + offset`) — NOT the `edge_keys()` tuple shape.
   It does NOT need to be updated for Option E (and must NOT be changed to the 8-field
   shape, which would break its own lookup). The only required change in the repro script
   is a comment update at lines 461–465 to reflect the new 8-field shape returned by
   `edge_keys()`.

7. **Rename probe script** from `test_56e3_option_e_probe.py` to
   `test_56e3_emission_ordinal_invariance.py` and update its module docstring to reflect
   permanent regression-guard status. Rationale: the probe's exact safety property is
   that same-`(canonical_render, source_module_id)` twins have read/write-matched ordinal
   flips — naming it `_regression` would label WHAT it guards, not WHAT it tests.
   `_emission_ordinal_invariance` names the invariant directly, making the guard's scope
   discoverable without reading the implementation.

8. **Run test suite** and verify the 7 56e3-affected tests now pass. Confirm n7og tests
   remain at 0 mismatches. Record the new FAILED+ERROR count.

---

## §6 Validation — expected failure delta and test classification

| Test | Pre-C3d | Post-C3d | Post-Option-E |
|---|---|---|---|
| `TestValidateSCCOverlap::test_gr_simple` | PASS | FAIL | PASS |
| `TestValidateSCCOverlap::test_gr_declaration_order` | PASS | FAIL | PASS |
| `TestValidateSCCOverlap::test_gr_interval` | PASS | FAIL | PASS |
| `TestValidateSCCOverlap::test_gr_noshadow` | PASS | FAIL | PASS |
| `TestValidateSCCOverlap::test_lws` | PASS | FAIL | PASS |
| `TestValidateSCCOverlap::test_gr_inc_together` | PASS | FAIL | PASS |
| `test_pack_before_swap_orderinverted` | PASS | FAIL | PASS |
| `test_negative_one_prev_iter_lr0_not_drained` | PASS | FAIL | PASS |
| `test_n7og_edge_keys_multifixture` (all 3) | PASS | PASS | PASS |
| `test_56e3_emission_ordinal_invariance` (formerly probe) | N/A | PASS | PASS |
| All other tests | unchanged | unchanged | unchanged |

**Failure count trajectory:** 27 FAILED + 2 ERROR → 19 FAILED + 2 ERROR (8 tests resurface: 6 SCC methods + 1 pack + 1 gr).

**Acceptance gate (in order):**
1. The 8 resurfaced tests all pass with the exact Failure types they pin
   (`OverriddenInputFailure` for SCC × 6, `OrderInvertedFailure` for pack-before-swap,
   `MissingWaitFailure` for lr-not-drained). No Failure-type re-pinning.
2. `test_n7og_edge_keys_multifixture` emits 0 mismatches on all 3 fixtures.
3. `test_56e3_regression` passes (probe: 0 `rw_differ` groups across all bodies).
4. No new failures introduced anywhere in the suite.

---

## §7 Probe script handling

**Keep as a permanent regression guard.** Rename to `test_56e3_emission_ordinal_invariance.py`.

The probe's `groups_rw_differ=0` result is the empirical foundation that makes Option E
safe. It establishes that no CMS schedule (on the canonical TF32+UsePLRPack fixtures)
reorders same-source-module same-render instruction twins into semantically distinct
positions. If this property breaks — due to a new CMS schedule that legitimately varies
operands within a same-render group — the probe fires before the change is committed.

Update the module docstring: replace "empirical probe" framing with "permanent regression
guard; see `56e3_OPTION_E_EMPIRICAL_PROBE.md` for the design rationale."

---

## §8 Risks and open questions

**R1. `source_module_id=None` in OTHER tests (not the 5 failing ones).**
Tests that do not need to distinguish producers (e.g., positive-path tests that assert
zero failures on well-formed captures) use `source_module_id=None` for all nodes. These
tests construct REF and SUBJ captures where every producer is unique by byte-key anyway
(no two producers write the same byte-key at the relevant consumer time), so the
`(None, 0)` producer identity does not cause collisions. These tests are unaffected.
However, any NEW test that builds a multi-producer scenario and relies on source-identity
discrimination must populate `source_module_id`.

**R2. `emission_ordinal` population in fixture builders.**
`assign_emission_ordinals` is already called by `make_capture`. If any test constructs
a `GraphNode` or `DataflowEdge` directly without going through `make_capture`, the
`emission_ordinal` field may be `-1` (sentinel). Such nodes would produce a key with
`emission_ordinal=-1`, which is unusual but hashable. Investigation found no such direct
construction in the currently-failing tests; this is a documentation risk, not a
blocking bug.

**R3. Future CMS schedules that vary operands in same-render+same-source groups.**
The probe guards this. If `groups_rw_differ > 0` appears for a new fixture, Option E
is unsafe for that fixture. The response is to investigate whether the ordinal-flip
swaps semantically distinct operations; if yes, the fix is a more refined discriminator
(e.g., add `canonical_render` back behind a source_module_id-is-None guard). This is
not anticipated based on the structural analysis in the probe memo.

**R4. `repro_cross_subiter_artifact.py` staleness.**
The standalone repro script is not in the test suite; it will not be caught by CI.
Updating it is hygiene only. If it falls out of sync and a developer runs it, they get
a wrong result. The plan includes updating it (step 6), but it is not a correctness
gate for the bead.

**`test_gr_inc_together` is in scope (resolved).**
All 6 SCC test methods use `_build_scc_clobber_pair` with `source_module_id=None`
on both producer and clobber. The collision analysis from §C applies identically to
`test_gr_inc_together`. It is included in the 8-test resurface count and the fixture
fix must cover it. The §6 table and failure-count prediction have been updated accordingly.
Confirm during validation run.

---

## §9 New beads to file

None. All findings in this plan are contained within the Option E scope. No new blockers
surfaced that require separate tracking.

The fixture `source_module_id` fix (§C / step 1) is a prerequisite sub-task within the
56e3 bead scope, not a separate bead. It is discovered work but not a blocker requiring
a new bead — it is implementable within this bead's commit.

---

**End of plan.**
