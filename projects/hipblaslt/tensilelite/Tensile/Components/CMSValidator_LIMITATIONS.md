# CMSValidator Known Limitations

> Trimmed by bead `yxr` audit at validator branch tip `e234c11f7a`. The
> "Architecture-Specific Behavior" section's per-pass enumeration was deleted
> because the named `verify_*` passes (`VERIFY_ASCENDING_ORDER`,
> `VERIFY_CORRECT_NUMBER_OF_INSTRUCTIONS`, `VERIFY_SCC_OVERLAP`,
> `_disable_cdna4_only_passes_for_gfx1151`) no longer exist —
> `TIMELINE_RULES` and `STRUCTURAL_RULES` in `CMSValidator.py` are now empty
> no-ops; validation is graph-native via `compare_graphs` /
> `validate_edge_wait_coverage` / `cumulative_issue_cycles` in
> `ScheduleCapture.py`. The forthcoming design doc (bead `2yg`) will absorb
> any architecture-specific commentary that needs to live alongside the
> live code. The multi-ISA / gfx1151 reactivation story is tracked in
> bead `l6q`.

## Unsupported Configurations

### DirectToLds = 0
- **Status**: Not supported
- **Error**: `assert kernel["DirectToLds"]` in Timeline constructor (`CMSValidator.py:935`)
- **Impact**: gfx1151 (RDNA 3.5) schedules currently bypass timeline construction; graph-native validation runs but DTL=0 operand rules are not yet wired (see bead `l6q`)

### LocalSplitU > 1
- **Status**: Not supported
- **Error**: `assert kernel.get("LocalSplitU", 1) == 1` in Timeline constructor (`CMSValidator.py:936`)

### UseF32XEmulation = True with UseDirect32XEmulation = False
- **Status**: Not supported
- **Error**: Raises `ValueError` in `_hook_up_packs_f32` and `hook_up_packs`
- **Workaround**: Set `UseDirect32XEmulation = True` when using TF32 emulation

### ForceUnrollSubIter with Register Reuse
- **Status**: Partially supported
- **Issue**: Does not fully account for VGPR reuse patterns in LR0/LR1/LR3 scheduling
- **Location**: `CMSValidator.py` (search for `ForceUnrollSubIter` — TODO at line ~990)

## Validation Gaps

### False Negatives (schedule appears valid but may fail)
1. **Quad-cycle estimation is conservative**: graph-side `cumulative_issue_cycles` (in `Tensile/Components/ScheduleCapture.py`) models MFMA execution latency and type-switch penalties cycle-exactly but does not account for SWaitCnt counter pressure or other dynamic stalls. Real cycle counts may be higher, making some schedules pass validation that would violate timing in practice. (Bead `8nz` deleted the structural-side mirror `estimate_quad_cycles`; the graph helper is the single source of truth.)
2. **SBarrier timing assumes instant synchronization**: The validator checks that an `s_barrier` exists between the required instructions but does not model barrier latency.
3. **SWaitCnt counter values are not verified**: The validator trusts that `dscnt`/`vlcnt`/`vscnt` values in `SWaitCnt` instructions are correct — it does not count actual outstanding memory operations to verify.
4. **VCC dataflow tracking is a workaround slated for removal** (bead `rocm-libraries-uraq`). Today `_VCCRule` (`ScheduleCapture.py:1436`) plus `_is_vcc` / `_vcc_resource()` / `_VCC_RESOURCE` / `_VCC_DST1_CARRY_OUT_CLASSES` populate VCC reads/writes on captured instructions so the dataflow graph can form VCC RAW edges. This is doubly redundant: q9j (bind `getDstParams`/`getSrcParams` to Python) gets the partition right at the rocisa level, and dzl (push implicit-operand metadata onto rocisa classes) adds the `regType="vcc"` the dataflow byte-key resolver needs. Until those land, the workaround stays — but it's marked for deletion. Once removed, **VCC RAW edges no longer form, VCC clobbers (`OverriddenInputFailure`) no longer fire, and same-body VCC reorders (`OrderInvertedFailure`) no longer fire**. Concrete tests that prove the gap: `test_vcc_carry_chain_reorder` and `test_vcc_rule_invariants` in `test_dataflow_graph_register_gaps.py`. Class of bug no longer caught after removal: 64-bit ALU adds reordering across the VCC carry chain, and any other VCC RAW dependency. **This entry will flip from "slated for removal" to "active limitation" when bead `uraq` lands**; q9j+dzl then close the gap.

### False Positives (schedule appears invalid but works)
1. None currently known.

## Architecture-Specific Behavior

### CDNA 4 (gfx950)
- Full graph-native validation via `compare_graphs` against the default schedule.
- `cumulative_issue_cycles` provides cycle-exact MFMA timing per CDNA 4 ISA section 7.6.
- 4x4 MFMA TF32 emulation pack chain is captured as graph nodes (CVT0 → 4x4 MFMAPack → CVT1 → MFMA).

### RDNA 3.5 (gfx1151)
- `Timeline` construction is bypassed (DTL=0 hits the `assert kernel["DirectToLds"]` guard at `CMSValidator.py:935`).
- Graph-native validation will run but DTL=0 operand rules and WMMA-shape rules are not yet wired in `ScheduleCapture._OPERAND_RULES`.
- See bead `l6q` for the planned multi-ISA reactivation.
- **Silence means "unchecked," not "correct"** for this ISA today.

### Older CDNA (gfx940, gfx941, gfx942, gfx90a)
- No CMS schedules registered for these ISAs.
- `hasCustomSchedule` returns `False`.

## idMap Limitations

- `idMap` is the source of truth for instruction-type resolution (`PACK_TYPE_MAP` in `CMSValidator.py`); it flows into `Timeline._populate_instructions` and into `ScheduleCapture.build_idmap` for graph-node construction.
- Tests must supply mock idMaps via `Tests/unit/cms_test_utils.make_mock_id_map()` (the `if "idMap" not in context` guard was removed in bead-stage 07).

## Ascending Order Exception

- Pack instructions (`PackA0`, `PackB0`, etc.) are excluded from the ascending-order check because pack ordering has its own validation via the graph-native dataflow comparison.
