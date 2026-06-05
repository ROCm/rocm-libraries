# Implementation Plan — C3a: UnrolledCapture Materializer + Tests

**Bead:** `rocm-libraries-abgv`
**Blocks:** `rocm-libraries-wg77` (C3b), `rocm-libraries-1rsy` (C3c)
**Depends on:** `rocm-libraries-5tf9` (C1, closed)
**File:** `Tensile/Components/ScheduleCapture.py` (primary)
**Test file:** `Tensile/Tests/unit/test_UnrolledCapture.py` (new)

---

## §1 Scope

Add `UnrolledCapture` and `UnrolledIterRecord` classes to `ScheduleCapture.py`, implement `UnrolledCapture.from_four_part_capture(fpc)` that materializes the unrolled timeline `PRO → ML_iter[0] → ML_iter[1] → NGL → NLL`, and add a unit-test file verifying the materialization contract. No downstream validator code is wired to consume `UnrolledCapture` in this commit. Validator state is unchanged from end-of-C1 (RED for the cross-subiter pack-MFMA edges exposed by the exemption deletion).

---

## §2 Investigation Findings

### A. FourPartCapture inventory

**File:** `Tensile/Components/ScheduleCapture.py:645–700`

```
FourPartCapture fields:
  main_loop: dict           # {codepath_int: LoopBodyCapture} — ML body per codepath
  main_loop_prev: dict      # {codepath_int: LoopBodyCapture} — ML-1 body per codepath
  n_gl: dict                # {0: LoopBodyCapture} or {} — NGL, always singleton
  n_ll: dict                # {0: LoopBodyCapture} or {} — NLL, always singleton
  num_mfma: int
  num_codepaths: int
  source: str               # 'cms' or 'default-sia3'
  num_mfma_per_subiter: int = 0
  arch_profile: Optional[ArchProfile] = None
  prologue: Optional[LoopBodyCapture] = None  # PRO body — None when PGR=0
```

Body label constants (`ScheduleCapture.py:711–723`):
- `BODY_LABEL_PROLOGUE = "PRO"` — maps to `loop_index = -1`
- `BODY_LABEL_ML_PREV = "ML-1"` — maps to `loop_index = 0`
- `BODY_LABEL_ML = "ML"` — maps to `loop_index = 1`
- `BODY_LABEL_NGL = "NGL"` — maps to `loop_index = 2`
- `BODY_LABEL_NLL = "NLL"` — maps to `loop_index = 3`

`_BODY_BUILD_ORDER` in `CMSValidator.py:1858`:
```python
_BODY_BUILD_ORDER = (BODY_LABEL_PROLOGUE, BODY_LABEL_ML_PREV, BODY_LABEL_ML, BODY_LABEL_NGL, BODY_LABEL_NLL)
```

**Key finding:** There is NO `BODY_LABEL_POST` or "POST" body in the existing code. The plan's §2.3 includes POST in the unrolled sequence, but it is not defined anywhere in `ScheduleCapture.py` or `CMSValidator.py`. C3a will **not** introduce POST — that body doesn't exist in the capture data model yet. The unrolled timeline for C3a is `PRO → ML_iter[0] → ML_iter[1] → NGL → NLL`. POST is reserved for a future commit when epilogue capture is added.

**Key finding on ML-1:** The plan's §2.3 says `ML_iter[k]` copies reuse the ML body (`main_loop`). `ML_MAT_COUNT = 2` means ML is copied twice. ML-1 (`main_loop_prev`) is NOT materialized as a separate ML iter — it is a distinct body that will be included in the unrolled sequence separately (once, like NGL/NLL). The bead scope says ML appears ML_MAT_COUNT times, not ML + ML-1 together. C3a materializes: `[PRO, ML_iter[0], ML_iter[1], NGL, NLL]` with ML-1 treated as a single body alongside the others. Since the bead description says the sequence is `PRO → ML_iter[0] → ML_iter[1] → ... → NGL → NLL → POST` and ML-1 is absent from this notation, ML-1 is not included in the C3a unrolled sequence. This is consistent with the NLL investigation §Q6 corrected shape which includes ML-1 separately, but the C3a scope matches the plan's bead description precisely.

### B. TaggedInstruction shape

**File:** `Tensile/Components/ScheduleCapture.py:413–551`

```python
@dataclass
class TaggedInstruction:
    wrapped: WrappedInstruction   # thin proxy around the rocisa instruction
    category: str                 # idMap category ("PackA0", "MFMA", "LRA0", etc.)
    slot: SlotKey                 # (subiter, slot_kind, mfma_index, sequence)
    emission_ordinal: int = -1    # per-(body, canonical_render, source_module_id) monotonic counter
    source_module_id: Optional[str] = None  # rocisa-derived named Module ancestor
```

Identity is computed by `identity_for(body_label: str) -> tuple` at line 475:
```python
return (WrappedInstruction.canonical_str(inst),   # canonical_render
        self.source_module_id,
        self.emission_ordinal)
```

**`emission_ordinal` is already present** (`ScheduleCapture.py:449`). No new fields need to be added to `TaggedInstruction` in C3a. The identity 3-tuple `(canonical_render, source_module_id, emission_ordinal)` already exists.

**CORRECTION (verifier 2026-06-05):** `identity_for`'s docstring at `ScheduleCapture.py:478` states `Format: (canonical_render, emission_ordinal)` — a stale 2-tuple description from before Approach 2 (rocm-libraries-dfd8). The actual return at lines 549-551 is `(canonical_render, source_module_id, emission_ordinal)`. This is a source-code docstring defect; the plan's design is correct. Test 13 must confirm the 3-tuple form against the actual return, not the stale docstring.

**Key finding:** `emission_ordinal` is a per-(body, canonical_render, source_module_id) counter assigned at finalize time (`assign_emission_ordinals` at line 778). When the ML body is materialized twice as iter copies, both copies share the same underlying `TaggedInstruction` objects — so `emission_ordinal` is identical across both copies by construction. Identity is iter-blind for free.

### C. ML_MAT_COUNT import

**Where it lives:** `CMSValidator.py:886`
```python
ML_MAT_COUNT = 2
```

**Current import direction:** `ScheduleCapture.py` does NOT import from `CMSValidator.py` at runtime (see module docstring, line 35: "this module is the upstream leaf"). Only a `TYPE_CHECKING`-guarded annotation import of `GraphNode` reaches back (line 62). `CMSValidator` imports from `ScheduleCapture` eagerly.

**Circular import risk:** If `ScheduleCapture.py` imports `ML_MAT_COUNT` from `CMSValidator.py` at runtime, it creates a circular import: `ScheduleCapture → CMSValidator → ScheduleCapture`. This would deadlock module initialization.

**Resolution:** Move `ML_MAT_COUNT = 2` from `CMSValidator.py` to `ScheduleCapture.py`. `CMSValidator.py` then imports it from there, which is already the established pattern for all other shared constants (all `BODY_LABEL_*`, `BODY_LABEL_TO_LOOP_INDEX`, `SchedulePosition`, `TaggedInstruction`, etc. live in `ScheduleCapture.py` and are imported by `CMSValidator.py`).

This is the least-disruptive placement:
- `ScheduleCapture.py` is the upstream leaf; it already owns all capture-structure constants.
- Moving the constant changes the import chain by one line in `CMSValidator.py` (from a definition to an import).
- No third file is affected — the constant is currently only defined in `CMSValidator.py` and not imported anywhere yet.
- Alternative (a shared constants module) is gratuitous new infrastructure for a single constant.

### D. UnrolledIterRecord shape

Each entry in the unrolled sequence is an `UnrolledIterRecord`. It must carry:

- `body_label: str` — which body this iter corresponds to (`"PRO"`, `"ML"`, `"NGL"`, `"NLL"`). Used as diagnostic annotation in downstream consumers.
- `iter_index: int` — 0 for non-ML bodies; `0..ML_MAT_COUNT-1` for ML iter copies.
- `instructions: List[TaggedInstruction]` — the ordered instruction list for this iter. For ML iter copies, this is the SAME Python list as the captured ML body's `instructions` (shared reference, not a copy). The `TaggedInstruction` objects themselves are shared; only `unrolled_position` stamps are distinct.
- `unrolled_start: int` — the `unrolled_position` of the first instruction in this record. Consumer code can compute per-instruction positions as `unrolled_start + stream_index_within_record` without storing per-instruction ints in the record itself.

The decision to share instruction objects (not copy them) is mandated by the plan: "ML_iter[k] reuses the underlying TaggedInstruction objects from the captured ML body." Sharing preserves the identity contract (`id(ti)` is the same across iter copies of the same instruction — which is the mechanism that makes `emission_ordinal` identical without any extra logic).

### E. unrolled_position semantics

**From UNROLLED_VALIDATION_PLAN.md §2.1:** "Iteration order matches stream execution order (PRO first, then ML's first iter copy, then ML's next iter copy, then NGL, then NLL, then POST)" and each GraphNode carries its `unrolled_position`.

`unrolled_position` is a monotonic integer across the entire unrolled timeline:
- Starts at 0 at PRO's first instruction.
- Increments by 1 per instruction, no body-boundary gaps.
- The unrolled stream has no position resets; every instruction position is globally unique.

This is confirmed by §3.2: "One linear pass. `latest_writer` initialized empty before PRO; updated by every write; queried by every read. No per-body or per-iter resets." The position is the index into the flat concatenated instruction list.

`unrolled_position` is an `int`, computed during materialization by maintaining a running counter across all bodies.

### F. Identity iter-blindness contract

**From UNROLLED_VALIDATION_PLAN.md §1 Constraints:** "Identity stays iter-blind 3-tuple; only its dict-keying role migrates" and "Pipelined instructions that move between iters must produce identical identity tuples so set-diff cancels them."

**Concretely:** `ML_iter[0]` and `ML_iter[1]` both reference the SAME `TaggedInstruction` object for instruction `i`. That object has one `emission_ordinal` (set at finalize time, body-scoped). `identity_for(body_label)` returns `(canonical_render, source_module_id, emission_ordinal)` — the same 3-tuple for both iter copies because both point to the same `TaggedInstruction`.

`unrolled_position` and `iter_index` are NOT part of `TaggedInstruction.identity_for(...)`. They are properties of where an iter copy appears in the unrolled timeline, not of the instruction's content. The downstream C3c commit will put `unrolled_position` on `GraphNode` (not on `TaggedInstruction`), and `GraphNode.identity` will remain `tagged_inst.identity_for(body_label)`.

Verification from §1: "the unrolled stream is a single position space" and "Each GraphNode carries its `unrolled_position`, the body-label annotation it came from, and (for ML iter copies) the iter index."

### G. Existing test patterns

**File:** `Tensile/Tests/unit/test_ScheduleCapture.py`

Conventions observed:
- Test classes named `TestXxx` grouping related tests with plain `test_*` methods.
- Fixtures built inline or via module-level helpers (`_make_body`, `_make_capture`, `_opaque_inst`).
- `_opaque_inst()` returns a `rocisa.instruction.SNop(waitState=0)` — the cheapest rocisa instruction with no dataflow.
- Real rocisa instructions (`MFMAInstruction`, `DSLoadB128`) used when real dataflow semantics are needed.
- `dataflow_fixtures.py` provides `make_lr`, `make_mfma`, `make_capture` helpers for body-level fixtures.
- Imports of `BODY_LABEL_*` constants from `ScheduleCapture`, `DataflowGraph` from `CMSValidator`.
- `pytest.raises` for exception contracts.
- No mocking framework; all tests wire real rocisa objects or lightweight stand-ins.
- The new test file must import `UnrolledCapture`, `UnrolledIterRecord`, `ML_MAT_COUNT` from `ScheduleCapture` and use the same `_make_body` / `FourPartCapture` fixture helpers.

---

## §3 Design

### Class stubs

```python
# ScheduleCapture.py additions

# Move ML_MAT_COUNT here from CMSValidator.py:
ML_MAT_COUNT: int = 2
# "Number of main-loop iter copies materialized in the unrolled timeline.
#  Hardcoded for PrefetchGlobalRead=2; the validator has only been empirically
#  verified for this prefetch depth. See UNROLLED_VALIDATION_PLAN.md §9 Q1."


@dataclass
class UnrolledIterRecord:
    """One body-slice in the unrolled timeline.

    For ML bodies, one record exists per iter copy (iter_index 0..ML_MAT_COUNT-1).
    For PRO / NGL / NLL, exactly one record exists (iter_index = 0).

    `instructions` holds the ordered TaggedInstruction list for this slice.
    For ML iter copies, the list is a SHARED REFERENCE to the captured ML
    body's instructions — the TaggedInstruction objects are not copied.
    Shared identity across copies is preserved by construction: the same
    Python object has the same `emission_ordinal` and `canonical_render`,
    so `identity_for(body_label)` returns the same 3-tuple regardless of
    which iter copy references it.

    `unrolled_start` is the unrolled_position of the first instruction in
    this record. Per-instruction unrolled positions are derived downstream
    as `unrolled_start + local_stream_index` (the index of the instruction
    within this record's ordered list, following slot lex sort order).
    """
    body_label: str                       # BODY_LABEL_* constant
    iter_index: int                       # 0 for non-ML; 0..ML_MAT_COUNT-1 for ML
    instructions: List[TaggedInstruction] # ordered per slot lex sort
    unrolled_start: int                   # global position of first instruction


@dataclass
class UnrolledCapture:
    """The unrolled instruction timeline for one FourPartCapture.

    Materializes the unrolled sequence:
        PRO → ML_iter[0] → ML_iter[1] → ... → ML_iter[ML_MAT_COUNT-1] → NGL → NLL

    PRO, NGL, and NLL each appear at most once (absent when the corresponding
    FourPartCapture body is None / empty dict). ML appears exactly ML_MAT_COUNT
    times (currently 2). Each ML iter record shares the underlying
    TaggedInstruction objects with the captured ML body.

    `records` is the ordered sequence of UnrolledIterRecord in unrolled timeline
    order. The total instruction count across all records equals the total
    unrolled_position span.

    `total_instructions` is the sum of len(r.instructions) across all records.
    Equivalent to: max(unrolled_start + len(instructions)) of the last record.
    """
    records: List[UnrolledIterRecord]
    total_instructions: int

    @classmethod
    def from_four_part_capture(cls, fpc: "FourPartCapture") -> "UnrolledCapture":
        """Materialize the unrolled timeline from a FourPartCapture.

        Sequence (in order):
          1. PRO — fpc.prologue (skipped if None)
          2. ML_iter[0..ML_MAT_COUNT-1] — fpc.main_loop[0].instructions, shared
             reference, ML_MAT_COUNT copies, distinct unrolled_start per copy
          3. NGL — fpc.n_gl[0] (skipped if empty)
          4. NLL — fpc.n_ll[0] (skipped if empty)

        Each body's instructions are ordered by slot lex sort
        (assign_stream_indices_for_body ordering) before materializing so
        the unrolled_position sequence is consistent with the stream-emission
        order the graph builder expects.

        Raises ValueError if fpc.main_loop has no codepath 0 entry (ML body
        is mandatory for a meaningful unrolled timeline).
        """
        ...
```

### Key design decisions encoded in the stubs

1. `instructions` in `UnrolledIterRecord` is a shared reference for ML iter copies (not a deepcopy). This is the plan's explicit requirement. Only the `unrolled_start` offset differentiates the two ML iter records.

2. `unrolled_start` on the record, not per-instruction `unrolled_position` stored on `TaggedInstruction`. `TaggedInstruction` has no `unrolled_position` field and C3a does not add one. The downstream C3c commit will add `unrolled_position` to `GraphNode` when it constructs nodes from the unrolled stream.

3. `from_four_part_capture` takes codepath 0 for the ML body (the default-side or CMS-side capture at codepath 0). The CMS codepath multiplicity is a codegen concern; the unrolled timeline is per-codepath and C3a materializes codepath 0.

4. The ordering of instructions within each record follows the same `(slot.mfma_index, slot.sequence)` lex sort that `assign_stream_indices_for_body` computes. This ensures the unrolled_position ordering matches the stream-emission order.

---

## §4 Step-by-step implementation order

1. **Move `ML_MAT_COUNT` from `CMSValidator.py` to `ScheduleCapture.py`.**
   - Remove the definition at `CMSValidator.py:882–886`.
   - Add it to `ScheduleCapture.py` after the `BODY_LABEL_TO_LOOP_INDEX` dict (around line 723).
   - Add an import of `ML_MAT_COUNT` to `CMSValidator.py`'s imports from `ScheduleCapture` (the import block at lines 48–52 of `CMSValidator.py`).
   - Verify: `grep "ML_MAT_COUNT" CMSValidator.py` shows only the import line; `grep "ML_MAT_COUNT" ScheduleCapture.py` shows the definition and the new class usage.

2. **Add `UnrolledIterRecord` dataclass to `ScheduleCapture.py`** after the body-label constants block.

3. **Add `UnrolledCapture` dataclass** with `from_four_part_capture` implementation.
   - The implementation walks:
     - PRO: if `fpc.prologue is not None`, create one `UnrolledIterRecord(body_label=BODY_LABEL_PROLOGUE, iter_index=0, instructions=sorted_instructions, unrolled_start=cursor)`.
     - ML: for `k in range(ML_MAT_COUNT)`, create `UnrolledIterRecord(body_label=BODY_LABEL_ML, iter_index=k, instructions=ml_instructions, unrolled_start=cursor)` where `ml_instructions` is the same sorted list object for all copies.
     - NGL: if `fpc.n_gl` has key 0, one record.
     - NLL: if `fpc.n_ll` has key 0, one record.
   - `cursor` advances by `len(instructions)` after each record.
   - Sort each body's instructions using the canonical lex key `(slot.mfma_index, slot.sequence)` — the same key `assign_stream_indices_for_body` uses internally (`ScheduleCapture.py:771-773`). **Do not call `assign_stream_indices_for_body` directly** — it returns a `{id(ti): stream_index}` dict, not a sorted list. The sort must be done explicitly: `sorted(body.instructions, key=lambda ti: (ti.slot.mfma_index, ti.slot.sequence))`. For ML iter copies, compute the sorted list once and reuse the same Python list object for all `ML_MAT_COUNT` records.

4. **Write `Tensile/Tests/unit/test_UnrolledCapture.py`** (see §6).

5. **Run tests** to confirm all new tests pass and no existing tests regress.

---

## §5 Validation

### Commands

```bash
# From the worktree root: tensilelite/
tox -e unit -- -x \
  --ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py \
  Tensile/Tests/unit/test_UnrolledCapture.py

# Full unit suite (confirm no regression):
tox -e unit -- -x \
  --ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py
```

### What must pass

- All tests in `test_UnrolledCapture.py` (new).
- Pre-existing tests in the unit suite: **no NEW failures introduced**. C1 left the suite in a RED state (cross-subiter pack-MFMA edge failures from the exemption deletion). Those pre-existing failures must still be present and must not worsen. C3a is purely additive; it must not introduce any failures that were not already failing at end-of-C1.

### What must NOT change

- Validator behavior from end-of-C1 state. `compare_graphs`, `build_dataflow_graph`, `diagnose_missing_edge` are untouched.
- `TaggedInstruction`, `FourPartCapture`, `LoopBodyCapture`, `DataflowGraph` — no field additions.
- `ML_MAT_COUNT` numeric value stays `2`; only its location moves.

### Regression sentinel

```bash
grep "ML_MAT_COUNT" Tensile/Components/CMSValidator.py
# Must show only an import line, not a definition.

grep "ML_MAT_COUNT" Tensile/Components/ScheduleCapture.py
# Must show the definition and UnrolledCapture usage.
```

---

## §6 Tests to add

File: `Tensile/Tests/unit/test_UnrolledCapture.py`

### TestUnrolledCaptureBasicShape

**Fixture:** minimal `FourPartCapture` with PRO, ML (codepath 0), NGL, NLL bodies each containing a small number of SNop-tagged instructions.

**Test 1 — record count:**
`from_four_part_capture(fpc)` with PRO + ML + NGL + NLL present produces `1 + ML_MAT_COUNT + 1 + 1 = 5` records. Assert `len(uc.records) == 5`.

**Test 2 — body label order:**
Assert `[r.body_label for r in uc.records] == [BODY_LABEL_PROLOGUE, BODY_LABEL_ML, BODY_LABEL_ML, BODY_LABEL_NGL, BODY_LABEL_NLL]`.

**Test 3 — ML iter_index values:**
The two ML records have `iter_index` values `0` and `1`. Non-ML records all have `iter_index == 0`.

**Test 4 — PRO absent when prologue is None:**
Build a `FourPartCapture` with `prologue=None`. Assert `len(uc.records) == ML_MAT_COUNT + 1 + 1` and no record has `body_label == BODY_LABEL_PROLOGUE`.

**Test 5 — NGL absent when n_gl is empty dict:**
Build with `n_gl={}`. Assert `BODY_LABEL_NGL not in [r.body_label for r in uc.records]`.

**Test 6 — NLL absent when n_ll is empty dict:**
Same for NLL.

### TestUnrolledCapturePositionMonotonicity

**Fixture:** PRO (3 instructions), ML (4 instructions), NGL (2 instructions), NLL (5 instructions).

**Test 7 — unrolled_start values:**
Assert `uc.records[0].unrolled_start == 0` (PRO).
Assert `uc.records[1].unrolled_start == 3` (ML_iter[0]).
Assert `uc.records[2].unrolled_start == 7` (ML_iter[1]).
Assert `uc.records[3].unrolled_start == 11` (NGL).
Assert `uc.records[4].unrolled_start == 13` (NLL).

**Test 8 — total_instructions:**
Assert `uc.total_instructions == 3 + 4 + 4 + 2 + 5 == 18`.

**Test 9 — derived per-instruction positions are strictly monotonic:**
Derive per-instruction positions as `r.unrolled_start + local_idx` for all records and all instructions. Assert the resulting sequence is strictly increasing with no gaps.

### TestUnrolledCaptureMLSharing

**Test 10 — ML iter copies share instructions list identity:**
`ml0 = uc.records[1]`, `ml1 = uc.records[2]`. Assert `ml0.instructions is ml1.instructions`. The list object is shared, not copied.

**Test 11 — TaggedInstruction objects are the same Python objects:**
Assert `all(ti0 is ti1 for ti0, ti1 in zip(ml0.instructions, ml1.instructions))`.

### TestUnrolledCaptureIdentityIterBlindness

**Fixture:** ML body with a single MFMA instruction whose `emission_ordinal` is assigned by `assign_emission_ordinals`. 

**Test 12 — identity is identical across ML iter copies:**
`ti = uc.records[1].instructions[0]` (ML_iter[0]).
`ti2 = uc.records[2].instructions[0]` (ML_iter[1]).
Assert `ti is ti2` (same object — proves identity trivially).
Assert `ti.identity_for(BODY_LABEL_ML) == ti2.identity_for(BODY_LABEL_ML)`.

**Test 13 — iter_index does not appear in TaggedInstruction.identity_for:**
Confirm the identity 3-tuple is `(canonical_render, source_module_id, emission_ordinal)`. No `iter_index`, no `unrolled_position`.

### TestUnrolledCaptureMLMatCount

**Test 14 — ML_MAT_COUNT drives ML copy count:**
Assert `sum(1 for r in uc.records if r.body_label == BODY_LABEL_ML) == ML_MAT_COUNT`.

**Test 15 — ML_MAT_COUNT is importable from ScheduleCapture:**
`from Tensile.Components.ScheduleCapture import ML_MAT_COUNT; assert ML_MAT_COUNT == 2`.

---

## §7 Open questions / risks

1. **POST body not in scope — bead description corrected.** `BODY_LABEL_POST` does not exist in the codebase, and POST is absent from both `FourPartCapture` and `_BODY_BUILD_ORDER`. The bead description's sequence `PRO → ML_iter[0] → ... → NGL → NLL → POST` included POST in error. The actual C3a unrolled sequence is `PRO → ML_iter[0..ML_MAT_COUNT-1] → NGL → NLL` only. UNROLLED_VALIDATION_PLAN.md §5 Commit 3a's test-coverage list also includes POST, which is similarly in error.

   **Resolution (verifier 2026-06-05):** Option (a) — POST has no capture infrastructure and does not belong in C3a. The bead description has been updated to remove POST from the materialized sequence. No new blocker bead is required: POST is absent from all C3b/C3c/… bead scopes as well, so no downstream commitment depends on POST appearing in C3a's materializer. If POST body capture is ever added, the materializer will need a new iter record, but that is future work.

2. **ML-1 (main_loop_prev) absent from the unrolled sequence.** ML-1 exists in `FourPartCapture.main_loop_prev` and in `_BODY_BUILD_ORDER`, but the bead description and UNROLLED_VALIDATION_PLAN.md §5 Commit 3a both omit it from the unrolled timeline. C3a does not materialize ML-1. The NLL investigation §Q6 suggests the correct shape is `PRO → ML-1 → ML → NGL → NLL`, but the bead scope explicitly uses ML iter copies (not ML-1) to model cross-iter live-ins. This design choice must be documented in C3c's plan before C3c wires the walk into `build_dataflow_graph`: C3c must either (a) emit ML-1 once before ML iter copies, or (b) explain why ML-1's live-outs are subsumed by the ML iter structure. This is a C3c design decision, not a C3a blocker — C3a's materializer cannot decide for C3c. File under C3c's pre-implementation checklist.

3. **Codepath 0 assumption.** `from_four_part_capture` takes `main_loop[0]` for the ML body. CMS captures may have multiple codepaths in `main_loop`. The single-codepath assumption is correct for the unrolled timeline (the unrolled walk models one execution path), but may need clarification in the docstring.

4. **Instruction ordering within records.** The plan mandates `(slot.mfma_index, slot.sequence)` lex sort, which is what `assign_stream_indices_for_body` computes. Production captures are already in this order (builder appends in slot order). Synthetic fixtures may not be. The `from_four_part_capture` implementation should re-sort to be safe. The sort is idempotent for well-ordered captures.

5. **`FourPartCapture` with no codepath 0 in main_loop.** The implementation should raise `ValueError` clearly. This shouldn't occur in production but is worth guarding in tests.

---

## §8 New beads to file

**CORRECTION (verifier 2026-06-05):** The original plan claimed no new beads were required. Two issues were resolved during verification:

1. **POST removed from bead scope.** The bead description for `rocm-libraries-abgv` listed POST in the materialized sequence. This was an error — POST has no capture infrastructure. The bead description has been updated (see §9 below) to remove POST from the sequence. No blocker bead is needed because no downstream C3b/C3c/… bead assumed POST would exist in the C3a materializer.

2. **ML-1 design gap deferred to C3c's plan.** ML-1 exists in `FourPartCapture` and `_BODY_BUILD_ORDER` but is absent from the C3a unrolled sequence. This is an intentional design choice in the bead spec (the bead explicitly uses ML iter copies, not ML-1, to model cross-iter live-ins). The question of whether C3c should emit ML-1 before the ML iter copies must be resolved in C3c's pre-implementation plan — not as a blocker bead, because C3c's bead already covers `build_dataflow_graph` rewrite and the ML-1 question is an in-scope design decision for that commit.

No new blocker beads are required at C3a time.

---

## §9 Bead description update (required before implementation)

The `rocm-libraries-abgv` bead description must be updated to remove POST from the materialized sequence. The corrected `description` field for `br update` is:

```
Scope:
1. Add UnrolledCapture and UnrolledIterRecord classes in ScheduleCapture.py.
2. Implement UnrolledCapture.from_four_part_capture(fpc) materializing the unrolled sequence:
   PRO -> ML_iter[0] -> ML_iter[1] -> ... -> ML_iter[ML_MAT_COUNT-1] -> NGL -> NLL
   PRO/NGL/NLL appear once each (absent when the corresponding FourPartCapture body is
   None / empty dict). ML appears ML_MAT_COUNT times (= 2). Each ML iter copy reuses the
   underlying TaggedInstruction objects but stamps a distinct unrolled_start and iter_index.
   POST body has no capture infrastructure yet and is excluded from this materializer.
3. Identity stays iter-blind: same canonical_render + source_module_id + emission_ordinal
   across iter copies.
4. Add unit tests verifying materialization: per-body counts, unrolled_position
   monotonicity, identity-iter-blindness contract.
5. NO validator code consumes UnrolledCapture yet — purely additive infrastructure.

Acceptance:
- UnrolledCapture reachable and tested in isolation.
- No new test failures beyond end-of-C1 RED state.
- Validator state unchanged from end-of-C1.
```

Run before implementing:
```bash
ACTOR="${BR_ACTOR:-assistant}"
br update --actor "$ACTOR" rocm-libraries-abgv --description "..."
br sync --flush-only
```
