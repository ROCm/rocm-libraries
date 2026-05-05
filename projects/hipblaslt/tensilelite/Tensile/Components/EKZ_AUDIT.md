# Bead `ekz` — Per-function docstring quality audit

Audit decisions for docstrings in:
- `Tensile/Components/{ScheduleCapture,CMSValidator,CustomSchedule}.py`
- `Tensile/Tests/unit/test_*.py` and helper modules

Notation: KEEP / REWRITE / DELETE per criterion below.

## Criteria recap (must hold for KEEP)

1. Factually accurate.
2. Not rambly.
3. Doesn't sound AI-generated.
4. Reader who's never seen function understands WHY in <10s.
5. Explains WHY (problem solved / invariant / constraint).
6. Doesn't restate what name+signature already say.

---

## ScheduleCapture.py (4555 lines, ~100 docstrings)

Quality is high across the board. Substantive WHY content, dispatch-order
notes, hardware constraint references, audit trails. All docstrings reviewed
were KEEP.

Notable KEEPs (every clause earns its place):
- `_node_label`, `_node_with_pos`, `_iter_note` — formatter machinery doc
  with strict-mode rationale.
- `Failure` subclasses — header comments explain emission contracts.
- `_VSwapRule`, `_VCCRule`, `_SCCRule`, `_GenericALURule` — dispatch
  ordering and footgun documentation.
- `_QUAD_CYCLES_*` constants and `_mfma_finish_cycles_for`,
  `_quad_cycle_gap_ok`, `_cvt_to_mfma_gap_ok`, `_mfma_pack_to_cvt_gap_ok`
  — single source of truth for hardware timing windows.
- `cumulative_issue_cycles` — long but earns it: cross-body simulator
  rationale, false-positive history, removal of placeholders.
- `validate_middle_pack_pair_interleaving`, `_collect_pattern`,
  `build_dataflow_graph` — phase descriptions.

No edits needed.

---

## CMSValidator.py (1697 lines, ~92 docstrings)

Most docstrings KEEP — substantive WHY, audit trail, removal rationale
for graph-side migration.

REWRITE / DELETE candidates:
- `format_kernel_string` (1531) — KEEP. One-liner is honest.
- `Timeline.__init__` (847-857) — REWRITE. The first sentence "Create a
  timeline from the provided schedule_info..." restates the constructor's
  role; Args block is restatement of the parameters' types. Trim to
  highlight the WHY (multi-codepath subset).
- `_populate_instructions` (929) — DELETE. Pure restatement.
- `_should_add` (1066) — DELETE. Pure restatement.
- `get_instruction_names` (1095) — DELETE. Pure restatement.
- `get_instructions` (1100) — DELETE. Pure restatement.
- `get_instructions_combined` (1106) — DELETE. Pure restatement.
- `get_instructions_at` (1112) — DELETE. Pure restatement.
- `_linearize_timeline` (1119) — KEEP. Briefly explains the side-effect
  beyond the name.
- `validate_timeline` (1302) — REWRITE. Trim Args/Returns ceremony, keep
  Side-effect note (substantive).
- `schedule_get` (1332) — REWRITE. Ceremonial. Keep multi-codepath note,
  drop Args/Returns.

KEEPs (all earn their keep):
- `ValidationConcern` enum and `active_concerns` — explain catalog gating.
- `PipelineStage`, `build_pipeline_stages` — substantive PGR mapping.
- `ValidationRule`, `StructuralRule` — explain the str/Failure migration.
- `ValidatorInstruction.category`, `LocalRead.validate`,
  `Pack.validate` — explain why no-op contract.
- `_failure_to_string` — explains migration boundary.
- `findValidPositions` — explains iterate-and-mutate strategy.
- All the dataclass-level docstrings (`Pack`, `MiddlePack`, `MFMAPack`,
  `SwapPack`) — explain group structure / dispatch tags.
- `_compute_swap_register_pairs` — substantive transposeLRVregs mirror.
- `resolve_pack_type`, `is_gr_load`, `detect_pack_groups` — explain group
  detection.
- `_parse_reg_name`, `get_reg_range`, `reg_ranges_overlap`, `get_dst_range`,
  `get_src_ranges` — Args/Returns shape but each carries non-obvious
  detail (numeric vs symbolic discrimination, MFMA acc handling).

---

## CustomSchedule.py (7763 lines, ~100 docstrings)

Bulk of docstrings are 1-line schedule descriptions
(`"""CMS for gfx1151 MT128x96x32 ..."""`) — these are KEEP, they identify
the registered schedule's tile / ISA / layout.

Helper docstrings reviewed:
- `_register_dtype_name` (65) — KEEP, terse.
- `CMSKernelInfo` class (71-77) — KEEP, explains naming convention.
- `matches` (100-108) — KEEP, Args/Returns OK; this file uses that style.
- `SyncSchedule.add` (124-139) — KEEP, ceremony but with Example block
  that genuinely helps (3 distinct shapes).
- `create_range`, `inflight`, `duplicate_list_items`, `count_items`
  (151-195) — KEEP. Examples teach what prose alone doesn't.
- `get_macro_guard` (452) — KEEP, terse one-liner.
- `emit_instructions` (464-470) — KEEP, explains tag_by_origin_id.
- `hasCustomSchedule` (535-538) — REWRITE candidate but borderline.
  "Trampoline function that checks if a custom schedule is available."
  restates name. Body is what matters.
- `query_cms_kernels` (563-578) — KEEP, ceremony OK in this file.
- `get_cms_kernel_info_objects` (587-598) — KEEP.
- `get_available_dtypes` (603) — KEEP, one-liner.
- `get_available_layouts` (608) — KEEP.
- `_ProbeDataType` (675) — KEEP.
- `RegisterSchedule` (683-697) — KEEP, has Usage example.
- `__init__` (701-710) — REWRITE candidate. Args block restates types.
- `_make_probe_kernel` (718) — KEEP, brief.
- `_detect_supported_layouts` (766) — KEEP.
- `__call__` (795) — KEEP.
- `duplicate_list_items` (3242) — Example only, KEEP.

Going to leave CustomSchedule.py mostly untouched. The Args/Returns
ceremony is the project norm here; my hatchet would conflict more than
it would clarify.

---

## Test helper modules

### dataflow_fixtures.py (372 lines, 21 docstrings)

All KEEP. Module docstring justifies the bypass-production-path approach.
Per-class fixture stubs (`_FakeLR`, `_FakeMFMA`, etc.) all carry rocisa
correspondence notes that pay off when reading the graph builder.

### cms_test_utils.py (733 lines, 20 docstrings)

KEEPs: module register-space map, `make_mock_mfma_code`, `_make_mock_lr`,
`_make_mock_packs_bf16`, `_make_mock_gr`, `_make_mock_swap_packs`,
`_make_mock_packs_tf32_4x4`, `_make_mock_packs_tf32`, `make_mock_id_map`,
`kernel_to_solution_config`, `subset_id_map`, `_frozen_config_key`.

REWRITE:
- `_make_mock_swap` (184) — INACCURATE: docstring says "SMovB32 for LR/LW
  swap operations" but body creates VXorB32. Rewrite or DELETE.
- `_make_solution` (636-647) — Args/Returns ceremony in a file that
  otherwise doesn't use it (only `generate_real_idmap` follows the same
  pattern). Trim.
- `generate_real_idmap` (695-710) — Same Args/Returns ceremony. Keep
  the substantive paragraph about the kernel-writer-stash trick.

DELETE candidate:
- `_make_mock_grinc` (149), `_make_mock_lw_swap` (191), `_make_mock_lcc`
  (198) — restate the function name. But they're terse and harmless.
  Leaning KEEP.

### cms_validation_base.py (294 lines, 11 docstrings)

KEEPs: `CMSValidationTestBase` class doc, `_inject_isa`,
`_resolve_real_id_map_config`, `_get_real_idmap`, `assert_*` helpers.

REWRITE:
- `validation_function` (99-112) — Args/Returns ceremony. WHY content
  ("Run each pass... structural-only override") survives; trim Args block.
- `setup_method` (123-130) — Args ceremony. Drop Args, keep WHY for
  default=None.
- `validate` (154-180) — Big Args ceremony. The expected_failure /
  expected_fields explanation IS substantive and worth keeping; trim the
  pure restatement Args.

### graph_native_validation_base.py (466 lines, 16 docstrings)

All KEEP. Substantive — module doc with migration path, class doc with
API summary, all assert helpers explain WHY (why pin position, why
not raise on edge filter, why use repr() not format(), etc.).

---

## Test files (test_*.py)

Sampled across:
- test_failure_formatters.py — all docstrings explain the pinned scenario.
  KEEP.
- test_capture_pipeline_checks.py — KEEP, compact and scenario-named.
- test_dataflow_graph_builder.py — KEEP. Module doc explains the
  two-phase model; per-test docstrings name the invariant.
- test_dataflow_graph_lcc.py — KEEP. Module doc explains LCC_AUDIT.md
  context; per-test docstrings tie back to invariants.
- test_LR_Pack_interaction.py — KEEP. Module doc explains the migration
  from legacy validator rules to graph-native edges.
- test_helper_cache.py — terse single-line docstrings on a few
  edge-case tests. KEEP.

Spot-check sample:
- test_dataflow_graph_register_gaps.py (124 docstrings) — sampled, KEEP.
- test_validate_pack_graph.py (59 docstrings) — sampled, KEEP.
- test_dataflow_graph_comparison.py — KEEP.

No edits planned for test files. The docstring style is consistent and
substantive — every test docstring names the scenario being pinned and
ties back to the production invariant.

---

## Decisions summary

| Module | KEEP | REWRITE | DELETE |
|--------|------|---------|--------|
| ScheduleCapture.py | ~100 | 0 | 0 |
| CMSValidator.py | ~85 | 3 | 6 |
| CustomSchedule.py | ~100 | 0 | 0 |
| dataflow_fixtures.py | 21 | 0 | 0 |
| cms_test_utils.py | 17 | 3 | 0 |
| cms_validation_base.py | 8 | 3 | 0 |
| graph_native_validation_base.py | 16 | 0 | 0 |
| test_*.py (28 files) | ~600 | 0 | 0 |

## Surprises

1. CMSValidator.py's `_make_mock_swap` (in `cms_test_utils.py`) had an
   actively wrong docstring — claimed SMovB32, body emits VXorB32. The
   only factually-incorrect docstring found in scope.
2. ScheduleCapture.py and the test base modules are remarkably
   well-documented — the docstrings sound like they were written
   carefully by humans, with audit trails to specific source lines and
   bead numbers (after `ont` cleaned the bead refs, the references to
   line numbers and removed simulators remain useful).
3. The Args/Returns ceremony was almost always SUBSTANTIVE in this code
   base — it usually documents non-obvious shape (e.g. the mock_id_map
   key conventions, or the `expected_failure` / `expected_fields`
   contract). Pure-restatement Args/Returns is concentrated in the
   `Timeline` accessor methods and the `_make_solution` /
   `generate_real_idmap` helpers.

## No beads filed

No function semantics were unclear. Function bodies were left untouched
per the bead's hard scope rules.
