# CMSValidator / ScheduleCapture Consolidation Audit (br4)

Audit produced for bead `rocm-libraries-br4`. Read-only inventory + sub-bead
plan for the future executor agent. **No code changes proposed inside this
document; only described.** Behavior must remain bit-exact at every sub-bead
boundary, and the 634-test suite must stay green.

For brevity I refer to the two files as `CMSValidator.py` (1482 lines) and
`ScheduleCapture.py` (5076 lines). All line numbers are against the worktree
tip at `users/alvasile/validator_long_term_plans` (HEAD = `7e57df7d0d`).

---

## Section 1 — Inventory of `CMSValidator.py`

### 1.A Module-level constants & enums

| Name | Lines | Verdict | Notes |
|---|---|---|---|
| `POSITION_INF`, `POSITION_NEG_INF` | 44–45 | KEEP-AS-IS | Sentinels consumed by `LocalRead.guaranteed_by`, `GlobalRead.needed_by`, `GlobalRead._validate_needed_by`. Live and load-bearing. |
| `ValidationConcern` enum | 47–68 | KEEP-AS-IS | Per-kernel coverage tracking; consumed by `active_concerns`, `isValid`. Live. |
| `ISA_CONCERN_CATALOG` | 72–97 | KEEP-AS-IS | Live; gfx950 + gfx1151 entries. |
| `MAIN_LOOP_PREV`, `MAIN_LOOP`, `NO_GLOBAL_LOAD_LOOP`, `NO_LOCAL_LOAD_LOOP` | 255–258 | KEEP-AS-IS | Used by `Timeline._populate_instructions`, `_should_add`. |
| `PACK_GROUP_SIZE_TF32`, `PACK_GROUP_SIZE_TF32_4X4` | 261–262 | KEEP-AS-IS | Used by `detect_pack_groups`. |
| `MFMAS_PER_TILE_TF32`, `MFMAS_PER_TILE_BF16` | 272–273 | DELETE | No callers anywhere in the worktree (grep returns only the definitions). Vestige. |
| `VGPRS_PER_CONVERSION_GROUP` | 276 | DELETE | Same — no callers. |
| `ALL_INSTRUCTION_NAMES` | 565–571 | KEEP-AS-IS | Used by `create_unified_timeline`. |
| `PACK_TYPE_MAP` | 577–588 | KEEP-AS-IS | Used by `resolve_pack_type`. |
| `GR_TYPE_MAP` | 592–598 | KEEP-AS-IS | Used by `is_gr_load`. |
| `TIMELINE_RULES` | 1298 | **DELETE** | Empty list. Iterated by `isValid` (1357, 1367) only to check `if any(...)` and skip. Removing the list also lets `isValid` drop its `timeline_needed` branch (1357–1374) and the entire timeline construction call. |
| `STRUCTURAL_RULES` | 1300 | **DELETE** | Same — empty list. Removing it also lets `isValid` drop the loop at 1342–1351. |
| `ValidatorPassContext` alias (1289) | 1289 | DELETE-CANDIDATE | Backward-compat alias. Grep shows zero callers using `ValidatorPassContext`; only `ValidationContext` is referenced. Safe to delete. |

### 1.B Functions

| Name | Lines | Verdict | Notes |
|---|---|---|---|
| `active_concerns` | 100–144 | KEEP-AS-IS | Live; called by `isValid`. |
| `build_pipeline_stages` | 165–191 | DELETE | No callers in repo. Authored speculatively for a PGR-N pipeline-model that never materialized; structural rule that consumed it has been removed. |
| `resolve_pack_type` | 601–621 | KEEP-AS-IS | Used by `detect_pack_groups`. |
| `is_gr_load` | 624–636 | KEEP-AS-IS | Used by `Timeline._populate_instructions`. |
| `detect_pack_groups` | 639–700 | KEEP-AS-IS | Used by `Timeline._populate_instructions`. |
| `_parse_reg_name` | 705–716 | DELETE | Used only by `get_reg_range` which is itself unreferenced (see below). |
| `get_reg_range` | 719–752 | DELETE | No callers in repo. The graph-side resolver uses `_byte_keys_for_resource` / `_reg_intersection` in ScheduleCapture.py instead; the structural-side rule that depended on this helper was deleted. |
| `reg_ranges_overlap` | 755–763 | DELETE | No callers — supplanted by `_reg_overlaps` / `_reg_intersection` in ScheduleCapture.py. |
| `get_dst_range` | 766–778 | DELETE | No callers. Same reason. |
| `get_src_ranges` | 781–802 | DELETE | No callers. Same reason. |
| `create_unified_timeline` | 805–824 | KEEP-AS-IS (gate on TIMELINE_RULES delete) | Only caller is `isValid` 1361. Once `TIMELINE_RULES` is empty and `isValid` no longer constructs a timeline (because `timeline_needed` is always False), this becomes test-only — `cms_validation_base.py:29` imports it directly. **Action:** keep — tests use it. |
| `applies_only_once` | 1107–1116 | DELETE | Decorator referenced by zero call sites. The pass machinery it was written for is gone. |
| `_compute_swap_register_pairs` | 1119–1161 | DELETE | No callers in repo. Authored to support the deleted swap-pack structural rule. |
| `_failure_to_string` | 1164–1185 | KEEP (move-with-isValid) | Helper for `validate_timeline` (1201) and `isValid` (1371). After `TIMELINE_RULES` is gone the only consumer is `validate_timeline`; could fold inline. Functionally live. |
| `validate_timeline` | 1188–1210 | KEEP-AS-IS | Live; `cms_validation_base.py` uses it for structural-rule-style timeline tests. Fires only the per-instruction `validate()` overrides on `ValidatorInstruction` subclasses (LocalRead/MFMA/Pack/etc.). |
| `schedule_get` | 1213–1220 | KEEP-AS-IS | Used by `Timeline._populate_instructions`. |
| `format_kernel_string` | 1303–1310 | DELETE | Zero callers in repo. |
| `isValid` | 1313–1423 | KEEP-AS-IS, MUST SHRINK | The live entry point. After `TIMELINE_RULES` / `STRUCTURAL_RULES` deletion, the per-code-path loops at 1340–1374 collapse to: just iterate `numCodePaths` for the pre-existing graph-comparison block (1380–1421), or remove the per-code-path loop entirely (graph comparison runs once). The mfma-reorder propagation at 1330 stays. |
| `findValidPositions` | 1426–1481 | KEEP-AS-IS | Live test helper / interactive diagnostic; calls `isValid`. |

### 1.C Classes

| Name | Lines | Verdict | Notes |
|---|---|---|---|
| `ValidationRule` (ABC) | 194–218 | DELETE | Only purpose was to type entries of `TIMELINE_RULES`. With the list gone, no subclasses exist. Grep confirms zero implementations. |
| `StructuralRule` (ABC) | 221–239 | DELETE | Same — typed `STRUCTURAL_RULES`. Zero implementations. |
| `PipelineStage` dataclass | 147–162 | DELETE | Returned by the also-dead `build_pipeline_stages`. Nothing consumes it. |
| `ValidatorInstruction` ABC | 280–315 | KEEP-AS-IS | Abstract base for `LocalRead`/`MFMA`/`Pack`/`GlobalRead`/`SWait`/`Barrier`/`SNop`/`GRInc`. Live (instantiated by `Timeline._insert`). |
| `LocalRead` | 318–337 | KEEP-AS-IS | `validate()` is now a no-op (correct: invariant moved graph-side); the override is preserved for the abstract-method contract. |
| `MFMA` | 340–342 | KEEP-AS-IS | No-op `validate()` for the contract. |
| `Pack`, `CVTPack`, `MiddlePack`, `SwapPack`, `MFMAPack` | 344–411 | KEEP-AS-IS | All `validate()` overrides are deliberate no-ops; keep for class taxonomy + isinstance dispatch. |
| `GlobalRead` | 414–510 | KEEP-AS-IS | Live `validate()` returns `MissingWaitFailure` / `MissingBarrierFailure`. The lazy import from ScheduleCapture (436) is the import-cycle pressure point — see Section 3. |
| `SWait` | 512–532 | KEEP-AS-IS | Live `validate()` returns `InvalidCounterValueFailure`. Lazy import (526). |
| `Barrier` | 534–544 | KEEP-AS-IS | Defensive sentinel-range check. |
| `SNop` | 546–555 | KEEP-AS-IS | Used by `_min_issue_quad_cycles_for` parity (ScheduleCapture.py 2957). |
| `GRInc` | 557–563 | KEEP-AS-IS | No-op `validate()` for the contract. |
| `Timeline` | 827–1104 | KEEP-AS-IS | Live; constructed by `create_unified_timeline`. Only consumer chain after `TIMELINE_RULES` death is `validate_timeline` (calls per-instruction `validate()`). |
| `ValidationContext` dataclass | 1223–1285 | KEEP-AS-IS | Live; passed to `isValid` and `validate_timeline`. The `default_capture` / `cms_capture` / `_last_failure` fields all have live consumers. |

### 1.D Summary

- **Pure deletions** (no behavior change, no imports to rewire): ~17 names totalling roughly 200–250 lines of dead code.
  - 5 dead constants (`MFMAS_PER_TILE_*`, `VGPRS_PER_CONVERSION_GROUP`, `TIMELINE_RULES`, `STRUCTURAL_RULES`, `ValidatorPassContext` alias)
  - 9 dead functions (`build_pipeline_stages`, `_parse_reg_name`, `get_reg_range`, `reg_ranges_overlap`, `get_dst_range`, `get_src_ranges`, `applies_only_once`, `_compute_swap_register_pairs`, `format_kernel_string`)
  - 3 dead classes (`PipelineStage`, `ValidationRule`, `StructuralRule`)
- **`isValid` body simplification** after the constant deletions: ~30–40 line shrink, no semantic change.
- **Everything else stays.** The `ValidatorInstruction` taxonomy + `Timeline` + `validate_timeline` is the live "structural per-instruction validation" surface; the typed `Failure` returns from `GlobalRead.validate` and `SWait.validate` are the live consumers of `cms_node_label` / `MissingWaitFailure` / `MissingBarrierFailure` / `InvalidCounterValueFailure`.

---

## Section 2 — Inventory of `ScheduleCapture.py`

The bead description requires moving "the NEW validation work" into
`CMSValidator.py`. The bulk of `ScheduleCapture.py` IS validate-time logic.

### 2.A Capture-time machinery (STAYS)

This is the legitimate residue if `ScheduleCapture.py` shrinks back to its
name.

| Name | Lines | Notes |
|---|---|---|
| `CaptureWiringError`, `CaptureSMEMError`, `CaptureFlatError`, `CaptureStoreError`, `CaptureMfmaCodeShapeError`, `CaptureConsistencyError`, `CaptureIdmapMismatchError`, `CaptureUnknownInstructionError`, `CaptureEmptyBodyError` | 59–101 | Capture-pipeline exception types. STAY. |
| `SLOT_KIND_PRE_LOOP`, `SLOT_KIND_MFMA`, `SLOT_KIND_POST_LOOP` | 108–110 | STAY. |
| `SlotKey` | 113–127 | STAY. |
| `WrappedInstruction` | 148–201 | STAY. |
| `MemoryRegion` | 204–225 | STRADDLES. Capture-time produces these (LDS slots), validate-time consumes them in `_intersection`. The dataclass itself is pure data — STAY in capture file. |
| `TaggedInstruction` | 228–249 | STAY. |
| `LoopBodyCapture` | 252–256 | STAY. |
| `CaptureContext` | 259–295 | STAY. Owned by KernelWriter. |
| `FourPartCapture` | 298–334 | STRADDLES. Holds `arch_profile` (validate-time concept) but is fundamentally the capture container handed to validators. STAY. The `arch_profile` field is fine here; it is only set by capture-time wiring. |
| `BODY_LABEL_*`, `BODY_LABEL_TO_LOOP_INDEX` | 523–533 | STRADDLES. Used both at capture time (`make_position`) and at validate time (`_BODY_BUILD_ORDER`, all the wait/barrier walkers). The constants themselves are pure naming — STAY in capture file; validators reference them via existing import. |
| `SchedulePosition` | 537–558 | STRADDLES — this is the key shared type. Imported eagerly by CMSValidator.py:39 today. The class itself fits in either file; leaving it here keeps the existing `from Tensile.Components.ScheduleCapture import SchedulePosition` working. STAY. |
| `make_position` | 561–571 | STAY (capture-time). |
| `LoopBodyCaptureBuilder` | 1090–1168 | STAY. The `finalize()` calls `_populate_wrapper` (operand rules) — see straddle note in §2.C below. |
| `split_for_plr`, `build_idmap`, `invert_idmap_to_id_to_category`, `structural_clone`, `build_id_to_category_per_iter`, `assert_idmap_completeness` | 1182–1430 | STAY. Capture-side schema builders / id-map utilities. |
| `clone_loop_body` | 4746–4756 | STAY. |
| `evaluate_guard`, `_value_if_expr`, `expand_cms_macro` | 4763–4916 | STAY. CMS-macro walker. Capture-side. |
| `kernel_emits_n_gl`, `kernel_emits_n_ll`, `assert_capture_body_consistency`, `build_cms_four_part_capture` | 4935–5076 | STAY. Capture-side body emission. |

### 2.B Validate-time logic (MOVE TO `CMSValidator.py`)

These do not depend on capture-pipeline state once a `FourPartCapture` /
`DataflowGraph` exists. They run on a fully-built graph. **Move all of this
to CMSValidator.py.**

| Name | Lines | Move kind | Notes |
|---|---|---|---|
| `ArchProfile` dataclass | 353–399 | RENAME | Validate-time timing constants. Pure dataclass. |
| `_DEFAULT_CDNA4_ARCH_PROFILE`, `_GFX1151_ARCH_PROFILE` | 406–472 | RENAME | Constants. |
| `_ARCH_PROFILES_BY_ISA` | 476–479 | RENAME | Lookup. |
| `_resolve_arch_profile_for_isa` | 482–491 | RENAME | Already imported by CMSValidator.py:1389. |
| `_resolve_arch_profile` | 494–512 | RENAME | Used by graph helpers. |
| `_PositionStr` | 679–701 | RENAME | Used by `cms_node_label`. |
| `_node_position` | 704–709 | RENAME | Free helper. |
| `FailureNodeLabel` | 712–738 | RENAME | Public Failure shape. **Imported by tests + by CMSValidator.py.** |
| `Failure` base + `_ordinal` | 670–676, 741–776 | RENAME | Public Failure shape. |
| `OrderInvertedFailure`, `MissingWaitFailure`, `WaitInsufficientFailure`, `MissingBarrierFailure`, `TimingTooCloseFailure`, `InvalidCounterValueFailure`, `OverriddenInputFailure` | 788–973 | RENAME | All public Failure subclasses. Already imported by CMSValidator.py and many tests. |
| `cms_node_label`, `_cms_iter_delta`, `_body_for_node` | 992–1064 | RENAME | Failure-label providers; used by `validate_edge_wait_coverage`, `diagnose_missing_edge`, and CMSValidator.py:436 (`GlobalRead._validate_needed_by`). |
| `GraphNode` dataclass | 574–603 | RENAME | Graph node. |
| `DataflowEdge` dataclass | 606–622 | RENAME | Graph edge. |
| `DataflowGraph` dataclass | 625–655 | RENAME | Graph container. Carries `arch_profile`; with ArchProfile moved, this moves too. |
| `PRODUCER_CATEGORIES_LDS`, `PRODUCER_CATEGORIES_GLOBAL`, `SWAIT_CATEGORY`, `SBARRIER_CATEGORY` | 1440–1444 | RENAME | Shared constants for graph walkers. |
| `counter_for` | 1447–1464 | RENAME | Used by graph walkers. |
| `_swait_drains` | 1467–1487 | RENAME | Used by walkers + edge-coverage classifier. |
| `_all_nodes_in_order` | 1490–1504 | RENAME | Reads `_BODY_BUILD_ORDER` (currently a constant in this file). After the move, `_BODY_BUILD_ORDER` should also move (it's stream-order ordering, used only by validators). |
| `waits_in_window`, `barriers_in_window` | 1507–1559 | RENAME | Validate-time walkers. |
| `_BODY_BUILD_ORDER` | 3023 | RENAME | One-line tuple. Validate-time. |
| `build_dataflow_graph` | 3026–3247 | STRADDLES — but mostly validate-time. Constructs the graph from a FourPartCapture; validators consume the graph. The function itself uses capture-side helpers (`_make_node`, `_collect_barrier_edges`). Move to CMSValidator.py and have it `import` the capture-side helpers it needs. The pure-function `_collect_barrier_edges` / `_collect_pattern` move with it. |
| `_collect_barrier_edges` | 3250–3289 | RENAME (with build_dataflow_graph). |
| `_collect_pattern` | 3292–3369 | RENAME (with build_dataflow_graph). |
| `compare_graphs` | 3396–3474 | RENAME. Public validator entry point. |
| `diagnose_missing_edge` | 3477–3800 | RENAME. Single-edge classifier. ~325 lines. |
| `_queue_depth_at`, `_producer_queue_position`, `_wait_drains_producer`, `_any_drains`, `_first_insufficient`, `_last_drain` | 3803–3879, 4391–4409 | RENAME. FIFO simulator used by both `diagnose_missing_edge` and `_classify_edge_coverage`. |
| `_QUAD_CYCLES_*` aliases, `_MFMA_TYPE_SWITCH_THRESHOLD_*` aliases | 3896–3914 | RENAME. Backward-compat module-scope aliases (some tests import them). |
| `_mfma_finish_cycles_for` | 3917–3960 | RENAME. |
| `_is_mfma_pack_producer`, `_is_mfma_producer`, `_is_cvt_pack_producer`, `_is_alu_producer` | 3963–4024, 3984–3997, 4000–4024, 4336–4388 | RENAME. Validate-time graph-node classifiers. |
| `cumulative_issue_cycles` | 4025–4210 | RENAME. The cycle simulator. ~185 lines. |
| `_quad_cycle_gap_ok`, `_cvt_to_mfma_gap_ok`, `_mfma_pack_to_cvt_gap_ok` | 4213–4333 | RENAME. The four pair-specific gap helpers (third lives at `_mfma_pack_to_cvt_gap_ok`). |
| `validate_edge_wait_coverage` | 4426–4467 | RENAME. Public entry. |
| `_classify_edge_coverage` | 4470–4631 | RENAME. ~160 lines. |
| `validate_middle_pack_pair_interleaving` | 4662–4743 | RENAME. Validate-time stream-shape invariant. |

**Total estimated move:** ~2200–2400 lines of validate-time code transitioning
from `ScheduleCapture.py` to `CMSValidator.py`. After the move
`ScheduleCapture.py` shrinks to roughly 2400–2500 lines (capture machinery
only); `CMSValidator.py` grows to roughly 3500–3700 lines.

### 2.C Operand rules (STRADDLES — the trickiest split)

The operand-rule registry classes (`_DSLoadRule`, `_DSStoreRule`,
`_DTLBufferLoadRule`, `_BufferLoadRule`, `_MFMARule`, `_NoDataflowRule`,
`_VSwapRule`, `_VCCRule`, `_SCCRule`, `_GenericALURule`) at 2230–2749, plus
their support helpers (`_is_lr` / `_is_lw` / `_is_gr` / `_is_mfma` / `_is_*`,
`_reg_signature`, `_get_param`, `_inst_*`, `_canonical_render`, `_class_tag`,
`_class_tag_from_category`, `_split_category_iter`, `_node_subiter`,
`_byte_keys_for_resource`, `_resolve_producers`, `_identity_for`,
`_OPERAND_RULES`, `_populate_wrapper`, `_reg_overlaps`, `_reg_intersection`,
`_memory_intersection`, `_intersection`, `_min_issue_quad_cycles_for`,
`_make_node`, `_NUMERIC_REG_FACTORIES`, `_ensure_numeric_factories`,
`_VCC_RESOURCE`, `_vcc_resource`, `_VCC_DST1_CARRY_OUT_CLASSES`,
`_SCC_SENTINEL`, `_get_scc_sentinel`, `_SCC_OPCODE_FLAGS`, the `_LR_/LW_/GR_/
MFMA_/SWAIT_/SBARRIER_/SNOP_/SSETPRIO_/CVT_PACK_/MIDDLE_PACK_CLASS_NAMES`
sets) all participate in BOTH:

- **Capture-time**: `LoopBodyCaptureBuilder.finalize()` (1167) calls
  `_populate_wrapper(ti.wrapped, category=ti.category)` to populate
  `wrapped.reads` / `wrapped.writes`.
- **Validate-time**: `build_dataflow_graph` uses `_make_node` /
  `_byte_keys_for_resource` / `_resolve_producers` / `_intersection` /
  `_min_issue_quad_cycles_for` to build the DataflowGraph.

**Recommended split:** Leave the entire operand-rule registry in
`ScheduleCapture.py` (it is honestly capture-shape data: "what does this
rocisa class read/write"). Validate-time code that needs `_intersection` /
`_byte_keys_for_resource` / `_make_node` imports them from ScheduleCapture
post-consolidation. This preserves the bead's invariant that
`CMSValidator.py` → `ScheduleCapture.py` is the one-way import direction.

`_make_node` calls `_min_issue_quad_cycles_for(inst, profile)` which depends
on `ArchProfile`. After ArchProfile moves to CMSValidator, `_make_node` would
need ArchProfile via parameter (which it already takes — the `profile`
argument). The function itself can either:

- **Option A**: Stay in capture file. CMSValidator passes the profile in.
  Capture file keeps `_min_issue_quad_cycles_for` (knows about SNop wait_state
  semantics — pure capture-time fact about an instruction's issue cost).
- **Option B**: Move `_make_node` and `_min_issue_quad_cycles_for` to
  CMSValidator. Capture file's `LoopBodyCaptureBuilder` no longer calls
  `_make_node` (it doesn't today — `_make_node` is only called by
  `build_dataflow_graph`). Only `_populate_wrapper` runs at capture time.

**Option B is cleaner** and is what the proposed sub-bead plan below assumes.
After the move:
- Capture file: `_populate_wrapper` + the operand-rule registry + the `_is_*`
  classifiers + `_reg_signature` + `_canonical_render` + `_byte_keys_for_resource`
  + `_intersection` + `_reg_intersection` + `_memory_intersection` +
  `_resolve_producers` + the `_VCC_RESOURCE` / `_SCC_SENTINEL` machinery.
  These are all "what does this rocisa class look like" — capture-shape facts.
- Validator file: `_make_node`, `_min_issue_quad_cycles_for`, `_class_tag`,
  `_class_tag_from_category`, `_split_category_iter`, `_node_subiter`,
  `_identity_for` — all operate on `GraphNode` / `TaggedInstruction.category`
  + ArchProfile, which are validator-side concepts.

### 2.D Dead code in ScheduleCapture.py

Grep across the worktree found no callers for the following — possible dead
code. The executor should sanity-check before deleting.

- None found at audit time. Every public symbol in ScheduleCapture.py has at
  least one importer (production or test). The hard work is moves, not
  deletions.

---

## Section 3 — Import topology before/after

### 3.A Before (current state)

```
KernelWriter.py
    ├── ScheduleCapture (CaptureContext, structural_clone, build_idmap,
    │                    LoopBodyCaptureBuilder, build_id_to_category_per_iter,
    │                    build_cms_four_part_capture, build_dataflow_graph,
    │                    compare_graphs, validate_edge_wait_coverage,
    │                    assert_capture_body_consistency, ...)
    └── (does not import CMSValidator directly)

CMSValidator.py
    ├── ScheduleCapture (eager: SchedulePosition)
    └── ScheduleCapture (lazy, inside method bodies):
        - MissingWaitFailure, MissingBarrierFailure, cms_node_label,
          _cms_iter_delta  (GlobalRead._validate_needed_by, line 436)
        - InvalidCounterValueFailure  (SWait.validate, line 526)
        - LoopBodyCapture  (_failure_to_string, line 1184)
        - build_dataflow_graph, compare_graphs, validate_edge_wait_coverage,
          _resolve_arch_profile_for_isa  (isValid, line 1381)

ScheduleCapture.py
    └── CMSValidator (TYPE_CHECKING-only): ValidatorInstruction (line 43)
        — pure type-hint, never resolved at runtime

CustomSchedule.py — does not appear in the imports we found, but per the
docstrings calls cmsv.isValid().

Tests — many import from BOTH files. Production hot path:
    KernelWriter ⟶ ScheduleCapture
    CustomSchedule ⟶ CMSValidator ⟶ ScheduleCapture
```

The lazy-import pattern in `CMSValidator.py` is the symptom: every public
Failure class, every graph-builder entry, lives in ScheduleCapture but is
imported back into CMSValidator function bodies to keep the runtime cycle
broken (CMSValidator imports SchedulePosition eagerly; ScheduleCapture would
have to import CMSValidator's structural-side dataclasses to even type-hint
its formatters, hence the TYPE_CHECKING guard at line 38–43).

### 3.B After (target state)

```
KernelWriter.py
    └── ScheduleCapture (capture-only): CaptureContext, structural_clone,
        build_idmap, LoopBodyCaptureBuilder, build_id_to_category_per_iter,
        build_cms_four_part_capture, assert_capture_body_consistency

CMSValidator.py
    └── ScheduleCapture (eager): SchedulePosition, FourPartCapture,
        LoopBodyCapture, BODY_LABEL_*, BODY_LABEL_TO_LOOP_INDEX,
        WrappedInstruction, TaggedInstruction, MemoryRegion,
        _populate_wrapper, _byte_keys_for_resource, _intersection,
        _is_mfma / _is_lr / _is_lw / _is_gr / _is_swait / _is_sbarrier /
            _is_snop / _is_ssetprio / _is_cvt_pack / _is_middle_pack,
        _canonical_render, _reg_signature, CaptureUnknownInstructionError,
        CaptureConsistencyError, CaptureEmptyBodyError,
        UnexplainedMissingEdgeError

ScheduleCapture.py
    └── (no longer imports anything from CMSValidator — drops the
         TYPE_CHECKING block. NodeLike type alias narrows to GraphNode.
         Wait — GraphNode moves out too. So NodeLike's GraphNode arm
         ALSO moves out, with cms_node_label / _node_position / etc.)
```

After the move the only direction of import is `CMSValidator` →
`ScheduleCapture`. The `TYPE_CHECKING` guard in ScheduleCapture.py:38–43
disappears (`NodeLike = Union["GraphNode", "ValidatorInstruction"]` moves
to CMSValidator.py — both classes live there).

### 3.C Cyclic-import hazards

| Hazard | Location | Sub-bead unit |
|---|---|---|
| `cms_node_label`, `_cms_iter_delta`, `_node_position`, `FailureNodeLabel` referenced by `GlobalRead._validate_needed_by` (CMSValidator.py:436) | These all live in ScheduleCapture today; CMSValidator imports them lazily. After moving `cms_node_label` and friends to CMSValidator, the lazy import disappears — `_validate_needed_by` becomes a same-file reference. | Untangled by the move itself. |
| `InvalidCounterValueFailure` referenced by `SWait.validate` (CMSValidator.py:526) | Same story. After the Failure-classes move, becomes a same-file reference. | Untangled by the move. |
| `LoopBodyCapture` referenced by `_failure_to_string` (CMSValidator.py:1184) | After consolidation, `_failure_to_string` lives in CMSValidator; `LoopBodyCapture` stays in ScheduleCapture; CMSValidator imports it eagerly. Already the safe direction. | Trivial. |
| `build_dataflow_graph` / `compare_graphs` / `validate_edge_wait_coverage` / `_resolve_arch_profile_for_isa` lazy-imported in `isValid` (CMSValidator.py:1381) | After the move, all four live in CMSValidator. Lazy block becomes direct reference. | Untangled by the move. |
| `_make_node` (in ScheduleCapture today, calls `_min_issue_quad_cycles_for(inst, profile)` with `ArchProfile`) | If `_make_node` and `_min_issue_quad_cycles_for` move to CMSValidator (Option B in §2.C), then `build_dataflow_graph` (which lives in CMSValidator) calls `_make_node` (also in CMSValidator) — same file. No cycle. | Designed-out by Option B. |
| `_populate_wrapper` (stays in ScheduleCapture) is called by `LoopBodyCaptureBuilder.finalize` (also in ScheduleCapture) | Same-file. No cross-file dependency. | Safe. |

**No actual cyclic-dependency hazards remain after the move** — the lazy
imports in CMSValidator.py exist precisely because of the inverted
dependency direction; once the validators move INTO CMSValidator, every
lazy import becomes a same-file reference or an eager import in the
documented direction.

---

## Section 4 — Proposed sub-bead plan

The order is: **deletions first** (lowest risk; no other code depends on
dead names); **easy moves second** (whole pure-function relocations whose
imports are already in the documented direction); **surgical splits last**
(operand-rule reshuffling, isValid restructuring).

Every sub-bead must keep all 634 tests green at its boundary.

---

### Sub-bead 1 — Delete `TIMELINE_RULES` + `STRUCTURAL_RULES` no-op shells

- **What it does:** Delete the two empty-list constants `TIMELINE_RULES`
  (CMSValidator.py:1298) and `STRUCTURAL_RULES` (CMSValidator.py:1300) plus
  their consuming loops in `isValid` (1340–1374). Also delete the unused
  abstract base classes `ValidationRule` (194–218), `StructuralRule`
  (221–239) — they had no remaining concrete implementations.
- **Files touched:**
  - `Tensile/Components/CMSValidator.py`
- **Tests it must keep green:** all 634; the loops being deleted are
  proven dead by the empty lists they iterate. No test asserts on the
  presence of either constant.
- **Estimated lines deleted:** ~80 (empty lists + abstract classes + the
  isValid loops + comments).
- **Dependencies:** none — completely independent.

### Sub-bead 2 — Delete dead constants and helper functions in CMSValidator.py

- **What it does:** Delete:
  - `MFMAS_PER_TILE_TF32`, `MFMAS_PER_TILE_BF16`,
    `VGPRS_PER_CONVERSION_GROUP` (lines 272–276) — zero callers.
  - `build_pipeline_stages` + `PipelineStage` (lines 147–191) — zero
    callers.
  - Unused register-helper utilities `_parse_reg_name`, `get_reg_range`,
    `reg_ranges_overlap`, `get_dst_range`, `get_src_ranges` (lines 705–802)
    — zero callers; supplanted by ScheduleCapture's `_reg_intersection` /
    `_byte_keys_for_resource`.
  - `applies_only_once` decorator (lines 1107–1116) — zero callers.
  - `_compute_swap_register_pairs` (lines 1119–1161) — zero callers.
  - `format_kernel_string` (lines 1303–1310) — zero callers.
  - `ValidatorPassContext` alias (line 1289) — zero callers.
- **Files touched:**
  - `Tensile/Components/CMSValidator.py`
- **Tests it must keep green:** all 634. None of the deletions has a
  caller.
- **Estimated lines deleted:** ~200.
- **Dependencies:** none.

### Sub-bead 3 — Move ArchProfile + arch resolvers to CMSValidator.py

- **What it does:** Move `ArchProfile` (ScheduleCapture.py:353–399),
  `_DEFAULT_CDNA4_ARCH_PROFILE` (406–416), `_GFX1151_ARCH_PROFILE`
  (442–472), `_ARCH_PROFILES_BY_ISA` (476–479),
  `_resolve_arch_profile_for_isa` (482–491), `_resolve_arch_profile`
  (494–512). Then in CMSValidator.py's `isValid` (line 1389), drop the
  lazy import block and call them directly.
  - `FourPartCapture.arch_profile` field stays on the capture-side
    dataclass (the type annotation switches to a forward reference or
    `Optional[Any]` to avoid an import cycle from ScheduleCapture into
    CMSValidator). Capture-side code never instantiates ArchProfile —
    it just stores whatever the validator passed in.
  - The legacy module-scope aliases `_QUAD_CYCLES_*` and
    `_MFMA_TYPE_SWITCH_THRESHOLD_*` (3896–3914) move WITH ArchProfile;
    they are derived from `_DEFAULT_CDNA4_ARCH_PROFILE` at module load.
- **Files touched:**
  - `Tensile/Components/ScheduleCapture.py` (delete, leave forward
    annotation on FourPartCapture)
  - `Tensile/Components/CMSValidator.py` (add)
  - `Tensile/Tests/unit/test_arch_profile_gfx1151.py` (re-point its
    imports — currently `from Tensile.Components.ScheduleCapture import
    ArchProfile, _DEFAULT_CDNA4_ARCH_PROFILE, _GFX1151_ARCH_PROFILE,
    _resolve_arch_profile_for_isa` per line 44)
- **Tests it must keep green:** all 634; particularly
  `test_arch_profile_gfx1151.py`. Likely additional consumers in
  `test_dataflow_graph_*.py` — grep `_resolve_arch_profile` /
  `ArchProfile` / `_DEFAULT_CDNA4_ARCH_PROFILE` and re-point.
- **Estimated lines moved:** ~180.
- **Dependencies:** Sub-bead 1 (gets `isValid` into a stable shape first).

### Sub-bead 4 — Move Failure classes + FailureNodeLabel to CMSValidator.py

- **What it does:** Move `_ordinal` (670–676), `_PositionStr` (679–701),
  `_node_position` (704–709), `FailureNodeLabel` (712–738), `Failure`
  base (741–776), and all subclasses `OrderInvertedFailure` (788–800),
  `MissingWaitFailure` (812–835), `WaitInsufficientFailure` (841–868),
  `MissingBarrierFailure` (874–896), `TimingTooCloseFailure` (902–917),
  `InvalidCounterValueFailure` (923–941), `OverriddenInputFailure`
  (959–973). Also move `cms_node_label` (992–1039), `_cms_iter_delta`
  (1042–1051), `_body_for_node` (1054–1064).
  - In CMSValidator.py, drop the two lazy imports inside
    `GlobalRead._validate_needed_by` (line 436) and `SWait.validate`
    (line 526) — they become same-file references.
- **Files touched:**
  - `Tensile/Components/ScheduleCapture.py` (delete; the `NodeLike` type
    alias at line 50 narrows to `Union["ValidatorInstruction"]` and is
    moved out — see step 5 — so this becomes a clean delete).
  - `Tensile/Components/CMSValidator.py` (add)
  - **Many test files**: `test_failure_formatters.py:40`,
    `test_graph_native_validation_base.py:34, 305`,
    `test_dataflow_graph_register_gaps.py` (multiple sites),
    `test_ScheduleCapture.py`, `test_LR_Pack_interaction.py`,
    `cms_validation_base.py` indirectly. Re-point all imports from
    `Tensile.Components.ScheduleCapture` to
    `Tensile.Components.CMSValidator`.
- **Tests it must keep green:** all 634, with particular attention to
  `test_failure_formatters.py` (asserts on Failure shape + wording).
- **Estimated lines moved:** ~330 (Failure-class definitions + label
  helpers).
- **Dependencies:** Sub-bead 3 (no required ordering — independent —
  but landing 3 first gets ArchProfile out of the way so the formatter
  comments referencing it are coherent).

### Sub-bead 5 — Move GraphNode / DataflowEdge / DataflowGraph to CMSValidator.py

- **What it does:** Move `GraphNode` (574–603), `DataflowEdge` (606–622),
  `DataflowGraph` (625–655). Move the `NodeLike` type alias (50) WITH
  `GraphNode` since it now lives near both arms.
- **Files touched:**
  - `Tensile/Components/ScheduleCapture.py` (delete)
  - `Tensile/Components/CMSValidator.py` (add)
  - Test files: `test_dataflow_graph_*.py`, `test_ScheduleCapture.py:39`
    (imports `GraphNode`), several others. Re-point.
- **Tests it must keep green:** all 634; the dataflow-graph test cluster
  exercises GraphNode/DataflowGraph extensively.
- **Estimated lines moved:** ~120.
- **Dependencies:** Sub-bead 4 (FailureNodeLabel/Failure must already be
  in CMSValidator so cms_node_label calls within the moved code don't
  jump back across the boundary).

### Sub-bead 6 — Move graph walkers + FIFO simulator to CMSValidator.py

- **What it does:** Move:
  - `PRODUCER_CATEGORIES_LDS`, `PRODUCER_CATEGORIES_GLOBAL`,
    `SWAIT_CATEGORY`, `SBARRIER_CATEGORY` (1440–1444).
  - `counter_for` (1447–1464), `_swait_drains` (1467–1487),
    `_all_nodes_in_order` (1490–1504), `waits_in_window` (1507–1541),
    `barriers_in_window` (1544–1559).
  - `_BODY_BUILD_ORDER` (3023).
  - `_queue_depth_at` (3803–3824), `_producer_queue_position`
    (3827–3844), `_wait_drains_producer` (3847–3875), `_any_drains`
    (3878–3879), `_first_insufficient` (4391–4399), `_last_drain`
    (4402–4409).
  - `_is_mfma_pack_producer` (3963–3981), `_is_mfma_producer`
    (3984–3997), `_is_cvt_pack_producer` (4000–4024),
    `_is_alu_producer` (4336–4388).
  - `_mfma_finish_cycles_for` (3917–3960).
  - `cumulative_issue_cycles` (4025–4210), `_quad_cycle_gap_ok`
    (4213–4251), `_cvt_to_mfma_gap_ok` (4254–4285),
    `_mfma_pack_to_cvt_gap_ok` (4288–4333).
  - `_class_tag` (1906–1941), `_class_tag_from_category` (1944–1995),
    `_split_category_iter` (2025–2034), `_node_subiter` (2037–2053),
    `_identity_for` (2122–2152), `_min_issue_quad_cycles_for`
    (2957–2997), `_make_node` (3000–3018).
- **Files touched:**
  - `Tensile/Components/ScheduleCapture.py` (delete a large block)
  - `Tensile/Components/CMSValidator.py` (add)
  - Tests that import any of these helpers (mostly
    `test_dataflow_graph_builder.py`, `test_dataflow_graph_register_gaps.py`).
- **Tests it must keep green:** all 634; high-risk sub-bead because
  many helpers move at once. Suggest running
  `pytest Tensile/Tests/unit/test_dataflow_graph_*` and
  `pytest Tensile/Tests/unit/test_validate_*` and
  `pytest Tensile/Tests/unit/test_cumulative_issue_cycles*` in CI before
  merge.
- **Estimated lines moved:** ~700.
- **Dependencies:** Sub-bead 5 (GraphNode must be in CMSValidator so
  `_make_node`'s return type makes sense).

### Sub-bead 7 — Move build_dataflow_graph (incl. _collect_barrier_edges) to CMSValidator.py

- **What it does:** Move `build_dataflow_graph` (3026–3247),
  `_collect_barrier_edges` (3250–3289), `_collect_pattern` (3292–3369).
  Each calls helpers from the previous sub-bead (`_make_node`,
  `_byte_keys_for_resource`, `_resolve_producers`, `_intersection`).
  After the move, `_byte_keys_for_resource`, `_resolve_producers`,
  `_intersection`, `_reg_intersection`, `_memory_intersection`,
  `_reg_overlaps` STAY in ScheduleCapture (they are intersection
  primitives consumed at capture time too — operand-rule registry uses
  them); CMSValidator imports them. Direction: CMSValidator →
  ScheduleCapture. ✓
- **Files touched:**
  - `Tensile/Components/ScheduleCapture.py` (delete)
  - `Tensile/Components/CMSValidator.py` (add)
  - Tests that import `build_dataflow_graph` directly (most graph
    tests). Re-point.
- **Tests it must keep green:** all 634; this is the central graph
  constructor.
- **Estimated lines moved:** ~360.
- **Dependencies:** Sub-bead 6 (helpers it calls).

### Sub-bead 8 — Move compare_graphs + diagnose_missing_edge to CMSValidator.py

- **What it does:** Move `compare_graphs` (3396–3474),
  `diagnose_missing_edge` (3477–3800).
  - In CMSValidator.py's `isValid`, drop the lazy import block at
    1381–1391 (everything imported there now lives in the same file).
- **Files touched:**
  - `Tensile/Components/ScheduleCapture.py` (delete)
  - `Tensile/Components/CMSValidator.py` (add; simplify `isValid`)
  - Tests importing these (`test_dataflow_graph_*.py`,
    `test_mfma_reorder_e2e.py`, others). Re-point.
- **Tests it must keep green:** all 634; particularly the cross-graph
  comparison tests (~50 tests).
- **Estimated lines moved:** ~430.
- **Dependencies:** Sub-bead 7 (graph constructor must already be in
  place).

### Sub-bead 9 — Move validate_edge_wait_coverage + _classify_edge_coverage to CMSValidator.py

- **What it does:** Move `validate_edge_wait_coverage` (4426–4467),
  `_classify_edge_coverage` (4470–4631),
  `validate_middle_pack_pair_interleaving` (4662–4743).
- **Files touched:**
  - `Tensile/Components/ScheduleCapture.py` (delete)
  - `Tensile/Components/CMSValidator.py` (add)
  - Tests importing these. Re-point.
- **Tests it must keep green:** all 634; particularly
  `test_validate_edge_wait_coverage*`,
  `test_validate_pack_graph*`,
  `test_dataflow_graph_register_gaps.py`.
- **Estimated lines moved:** ~250.
- **Dependencies:** Sub-bead 8.

### Sub-bead 10 — Final cleanup pass

- **What it does:**
  - In `ScheduleCapture.py`, drop the `TYPE_CHECKING` import block
    (38–43) — `ValidatorInstruction` no longer needs to be referenced
    here because the Failure formatters that referenced it have moved.
    The `NodeLike` type alias also moved out with `GraphNode`.
  - In `CMSValidator.py`, fold `_failure_to_string` (1164–1185) into
    its only caller `validate_timeline` (1188–1210) if convenient —
    optional cleanup.
  - Delete the comment block in CMSValidator.py:244–253 ("Pack-related
    invariants live graph-side") — after consolidation the references
    are now same-file, the cross-file pointer comment is stale.
  - Delete the comment block in CMSValidator.py:264–269 ("Quad-cycle
    visibility verdicts live graph-side") — same.
  - Verify with `grep -rn 'from Tensile.Components.CMSValidator' ...`
    that nothing in `ScheduleCapture.py`'s import chain reaches back
    into CMSValidator. Document the verified one-way edge in a
    one-line module docstring update.
- **Files touched:**
  - `Tensile/Components/ScheduleCapture.py` (small cleanups)
  - `Tensile/Components/CMSValidator.py` (small cleanups)
- **Tests it must keep green:** all 634.
- **Estimated lines deleted:** ~30.
- **Dependencies:** Sub-bead 9 (the final move).

---

**Total sub-bead count: 10.**

Sub-beads 1, 2 are pure deletions (no imports to rewire). Sub-beads 3–9
are moves; each rewires test + production imports. Sub-bead 10 is hygiene.

---

## Section 5 — Risks called out

These are observations from the audit that are not themselves sub-beads but
the executor must be aware of.

### 5.1 The operand-rule registry is the largest STRADDLE

The operand-rule registry (`_DSLoadRule` … `_GenericALURule` plus
`_OPERAND_RULES`, `_populate_wrapper`, the `_is_*` discriminator helpers,
the `_inst_*` field extractors, the `_VCC_RESOURCE` / `_SCC_SENTINEL`
machinery, the `_NUMERIC_REG_FACTORIES` factory map) participates in BOTH
sides:
- **Capture-time consumer**: `LoopBodyCaptureBuilder.finalize` →
  `_populate_wrapper(ti.wrapped, ti.category)` populates `wrapped.reads` /
  `wrapped.writes` so subsequent graph construction can read them.
- **Validate-time consumer**: `build_dataflow_graph` (and the helpers it
  calls) read `wrapped.reads` / `wrapped.writes` during edge formation.

The audit recommends the **operand-rule registry STAYS in
`ScheduleCapture.py`**. The "what does this rocisa class look like in terms
of reads/writes" knowledge is honestly capture-shape data. CMSValidator
imports `_byte_keys_for_resource`, `_intersection`, `_resolve_producers`,
`_reg_intersection`, `_memory_intersection`, `_reg_overlaps`,
`_populate_wrapper` (if needed), and the `_is_*` family from
ScheduleCapture in the documented direction.

### 5.2 The `Failure` dataclass + subclasses are imported by EVERY graph test

`grep "from Tensile.Components.ScheduleCapture import"` shows
~25 test files importing Failure subclasses. Sub-bead 4 must rewire all of
them in one PR (or do it in two passes: re-export from ScheduleCapture as a
deprecation shim, land sub-bead, then a follow-up cleanup). I recommend
doing it atomically — the test renames are mechanical (`from Tensile.
Components.ScheduleCapture import` → `from Tensile.Components.CMSValidator
import`) and a re-export shim adds clutter.

### 5.3 `cms_validation_base.py` and `test_failure_formatters.py` are the canonical Failure-shape consumers

- `cms_validation_base.py:29` imports `create_unified_timeline`,
  `ValidationContext`, `validate_timeline` from CMSValidator. Sub-bead 4
  doesn't touch these; sub-beads 1–2 might (Sub-bead 1 deletes the
  `STRUCTURAL_RULES`/`TIMELINE_RULES` shells but `validate_timeline` survives).
- `test_failure_formatters.py:40` is the single source of truth for Failure
  message wording. Sub-bead 4 MUST keep this test green; if the move
  changes module identity in any error rendering, regenerate the
  expected strings only after confirming the rendering is byte-identical.

### 5.4 `_resolve_arch_profile` reads `getattr(carrier, "arch_profile", None)`

`_resolve_arch_profile` (494–512) accepts a `DataflowGraph`, a
`FourPartCapture`, or a `GraphNode` (via its captured graph back-ref). After
Sub-bead 3 moves ArchProfile to CMSValidator, the resolver also moves. The
ArchProfile instance the FourPartCapture carries must remain valid; the
resolver only does an `isinstance(profile, ArchProfile)` check, so as long
as both files refer to the same class object (one canonical definition in
CMSValidator), it works. Forward type-references in
`FourPartCapture.arch_profile: Optional[ArchProfile] = None` (line 334)
need to switch to `Optional[Any]` or to a TYPE_CHECKING import in
ScheduleCapture (this would re-introduce a TYPE_CHECKING import IN THE
ALLOWED DIRECTION — capture imports validator types under TYPE_CHECKING
only — but it's still TYPE_CHECKING. The cleaner alternative is
`Optional[Any]` with a docstring.).

### 5.5 `clone_loop_body` must continue to deepcopy WrappedInstruction correctly

`clone_loop_body` (4746–4756) calls `deepcopy(body)`. The `WrappedInstruction.__deepcopy__`
(189–195) explicitly clones `reads` and `writes`. None of the sub-beads
touch this code, but since `wrapped.reads` may contain MemoryRegion or
RegisterContainer references — and after Sub-bead 3 those references stay
in ScheduleCapture — the deep-clone behavior remains correct.

### 5.6 `KernelWriter.py` does not import from CMSValidator (only from ScheduleCapture)

This is a pre-existing constraint and HOLDS post-consolidation. KernelWriter
calls `cmsv.isValid` indirectly via `CustomSchedule.py` (the schedule
producer); the validator entry is reached through the schedule's
`validate()` plumbing. The consolidation does NOT need to change
KernelWriter's import surface.

### 5.7 GFX1151_AUDIT/ probes import DataflowGraph

Two audit probes (`Tensile/Components/GFX1151_AUDIT/lr_divergence_capture_probe.py:148`
and `validator_coverage_probe.py:235`) import `DataflowGraph` from
ScheduleCapture. Sub-bead 5 (DataflowGraph move) must update them. They are
not part of the 634-test suite (one-shot diagnostic scripts), but they
should still resolve.

### 5.8 The `_class_tag_from_category` carve-outs are mature; don't tweak

`_class_tag_from_category` (1944–1995) handles the TF32 PackMFMA quirk
(MFMAInstruction objects categorized as `PackA*`/`PackB*`). The list of
prefix matches is hand-tuned and exercised by ~20 tests. Sub-bead 6 moves
it verbatim; do NOT take the opportunity to "clean it up" in the same PR.

### 5.9 The `cumulative_issue_cycles` simulator (185 lines) carries cross-body simulator state

The function uses tightly-coupled mutable state (`current_issue`,
`mfma_free_at`, `last_mfma_class`, `last_mfma_issue`) across body
boundaries. Sub-bead 6 moves it verbatim. Do NOT refactor inside the move
PR; if a refactor is wanted, file it as a separate bead post-consolidation.

### 5.10 `LCC_AUDIT.md` documents an LCC-related quirk in `build_dataflow_graph`

The "LCC instructions ARE included" note in `build_dataflow_graph`'s
docstring (3034–3037) plus the discussion at line 3140–3143 reflects a
real semantic decision (LCC issue cycles count toward
`cumulative_issue_cycles`). Sub-bead 7 moves the function verbatim; the
docstring travels with it.

### 5.11 PR sequencing for tests

Test files import from BOTH `CMSValidator` and `ScheduleCapture`. After
each MOVE sub-bead the affected tests' import lines change. Each sub-bead
must include the test-import updates atomically — landing the move PR
before the test-rewire PR breaks the build for one revision.

### 5.12 The `from Tensile.Components.ScheduleCapture import SchedulePosition`
at CMSValidator.py:39 is the ONLY eager top-level cross-file import in
either direction today

After consolidation the eager-import surface grows substantially
(CMSValidator imports ~20+ symbols from ScheduleCapture eagerly). Verify
no circular import surfaces during sub-bead landing:
`python -c 'from Tensile.Components import CMSValidator, ScheduleCapture'`
must succeed in either order on a clean Python.

---

## Section 6 — Out of scope (deliberate)

Per bead description and the "Out of scope" section therein, these are
acknowledged as separate work and not addressed here:

- **Renaming `CMSValidator.py` itself** — out of scope. Naming pass is
  bead `09y`.
- **Renaming `ScheduleCapture.py` itself** — out of scope. Same naming
  pass.
- **Behavior changes** (new Failures, deleted Failures, changed Failure
  message wording, threshold tweaks) — out of scope. Every Failure
  emitted today is bit-exact preserved by every sub-bead boundary.
- **Adjacent refactors** — `nn0` (methods-on-classes), `c70` (Register
  abstraction). Both can land independently of this consolidation in
  either order.
- **Updating the design doc (`2yg`) to reflect the consolidated
  surface** — done as a follow-up after this epic lands; not a sub-bead.
- **Updating `AGENT.md` (`ow1`) post-consolidation** — same.
