# Validator Test Deficiencies Audit (rocm-libraries-86wq)

**Branch base**: `users/alvasile/validator_long_term_plans` @ `bb96c76a19` (wx9.3 Phase 2+3 tip).
**Audit branch**: `users/alvasile/test-audit-86wq`.
**Baseline pytest** (pre-audit, `--ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py`): **672 passed / 2 skipped / 1 xfailed**.
**Post-audit pytest**: **672 passed / 2 skipped / 1 xfailed** (unchanged).
**Date**: 2026-05-08.

---

## Part A — Minimization summary

All minimizations are confined to `Tensile/Tests/unit/test_dataflow_graph_register_gaps.py`. The pattern that recurred in this file is the "realistic GEMM frame": a `make_lr(...)` + `make_swait(...)` pair (and sometimes a trailing `make_mfma(...)`) wrapped around the actual property under test. Because `cumulative_issue_cycles` slices the captured stream from producer to consumer, any LR/SWait that sits BEFORE the producer contributes nothing to the gap arithmetic the test pins; the same is true of MFMAs that come after the consumer but don't intersect any asserted edge. Once `_GenericALURule` (wx9.4.4) and the wx9.3 phase-3 operand-slot edge identity landed, the scaffolding became dead weight.

| File | Test | Instructions removed | Count |
|------|------|----------------------|-------|
| test_dataflow_graph_register_gaps.py | `TestLRSAddrChain::test_lrs_vxor_before_lr_invisible` | trailing SWait + MFMA in REF and SUBJ | 4 |
| test_dataflow_graph_register_gaps.py | `TestLWSAddrChain::test_lws_vxor_before_lw_invisible` | trailing SWait + MFMA in REF and SUBJ | 4 |
| test_dataflow_graph_register_gaps.py | `TestGRIncSRDChain::test_grinc_srd_waw_before_buffer_load` | trailing SWait + MFMA in REF and SUBJ | 4 |
| test_dataflow_graph_register_gaps.py | `TestPackRAW::test_pack_cvt_raw_from_lr` | trailing SWait + MFMA in REF and SUBJ | 4 |
| test_dataflow_graph_register_gaps.py | `TestDTLm0Tracking::test_dtl_m0_update_before_buffer_load` | trailing SWait + MFMA in REF and SUBJ | 4 |
| test_dataflow_graph_register_gaps.py | `TestDTLm0Tracking::test_dtl_m0_add_update_before_buffer_load` | trailing SWait + MFMA in REF and SUBJ | 4 |
| test_dataflow_graph_register_gaps.py | `TestMFMASelfRAW::test_mfma_acc_chain_reorder` | leading LR + SWait in REF and SUBJ | 4 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_mfma_acc_chain_same_slot_passes_under_exact_arithmetic` | leading LR + SWait | 2 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_mfma_acc_chain_consecutive_slots_no_failure` | leading LR + SWait | 2 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_mfma_acc_chain_slot_delta_2_independent_of_num_mfma_per_subiter` | leading LR + SWait (in helper) | 2 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_mfma_acc_chain_cross_body_uses_unified_simulator` | leading LR + SWait | 2 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_mfma_acc_chain_cross_body_strict_when_graph_missing` | (none — already minimal) | 0 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_mfma_acc_chain_diagnose_missing_edge_dispatches_through_mfma_branch` | leading LR + SWait in REF and SUBJ | 4 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_mfma_to_alu_consumer_zero_gap_emits_timing_too_close` | leading LR + SWait | 2 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_mfma_acc_chain_diagnose_missing_edge_dispatch_no_failure` | leading LR + SWait in REF and SUBJ | 4 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_mfma_acc_chain_just_meets_quad_cycle_gap` | leading LR + SWait (in `_build` helper, applied to both REF and SUBJ) | 2 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_mfma_to_mfma_cross_subiter_routing_exact` | leading LR + SWait (in `_build_cap` helper) | 2 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_mfma_producer_multi_consumer_varied_gaps_exact` | leading LR + SWait | 2 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_mfma_quadcycle_mutation_smell` | leading LR + SWait | 2 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_mfma_pack_acc_chain_4x4_to_standard_exact_gap` | leading LR + SWait | 2 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_mfma_pack_acc_chain_meets_finish_1_no_failure` | leading LR + SWait | 2 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_mfma_type_switch_standard_to_4x4_adds_one_to_actual` | leading LR + SWait | 2 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_mfma_type_switch_4x4_to_standard_adds_one_to_actual` | leading LR + SWait | 2 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_mfma_same_class_chain_no_type_switch_penalty` | leading LR + SWait in two captures | 4 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_mfma_pack_to_cvt1_zero_gap_emits_timing_too_close` | leading LR + SWait | 2 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_mfma_pack_to_cvt1_meets_5_cycle_gap_no_failure` | leading LR + SWait | 2 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_mfma_pack_to_cvt1_routed_to_pack_to_cvt_branch_not_quadcycle` | leading LR + SWait | 2 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_mfma_pack_to_cvt1_three_short_pins_actual_3` | leading LR + SWait | 2 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_mfma_pack_to_cvt1_one_short_pins_actual_4` | leading LR + SWait | 2 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_cvt_pack_to_mfma_zero_gap_emits_timing_too_close` | leading SWait (LR is producer) | 1 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_cvt_pack_to_mfma_meets_2_cycle_gap_no_failure` | leading SWait (LR is producer) | 1 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_cvt_pack_routed_to_quadcycle_not_alu` | leading SWait (LR is producer) | 1 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_cvt_pack_to_mfma_one_short_pins_actual_1` | leading SWait (LR is producer) | 1 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_cvt_pack_to_pack_alu_consumer_takes_alu_exemption` | leading SWait (LR is producer of cvt0's inputs) | 1 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_cvt_to_mfma_cross_body_ml_prev_to_ml_real_cycle_count` | leading SWait in ML-1 producer body (LR is producer) | 1 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_cvt_to_mfma_cross_body_ml_to_ngl_real_cycle_count` | leading SWait in ML producer body (LR is producer) | 1 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_cvt_to_mfma_cross_body_below_threshold_fires_failure` | leading SWait in ML-1 producer body (LR is producer) | 1 |
| test_dataflow_graph_register_gaps.py | `TestMFMAQuadCycleGap::test_cvt_to_mfma_same_body_old_vs_new_formula_documented_divergence` | leading SWait (LR is producer) | 1 |
| test_dataflow_graph_register_gaps.py | `_build_cross_body_pack_to_cvt_capture` (helper, indirectly minimizes 4 cross-body tests) | leading LR + SWait | 2 |
| test_dataflow_graph_register_gaps.py | `TestCumulativeIssueCycles::test_chain_with_intervening_alu_accumulates_issue_cycles` | leading LR + SWait | 2 |
| test_dataflow_graph_register_gaps.py | `TestCumulativeIssueCycles::test_chain_with_multiple_typeswitches_accumulates_stalls` | leading LR + SWait | 2 |
| test_dataflow_graph_register_gaps.py | `TestCumulativeIssueCycles::test_chain_with_typeswitch_above_threshold_no_stall` | leading LR + SWait | 2 |
| test_dataflow_graph_register_gaps.py | `TestSNopWaitState::test_snop_wait_state_shifts_consumer_issue` (parametrized) | leading LR + SWait | 2 |
| test_dataflow_graph_register_gaps.py | `TestSNopWaitState::test_snop_wait_state_above_finish_dominates_standard_mfma` | leading LR + SWait | 2 |
| test_dataflow_graph_register_gaps.py | `TestSNopWaitState::test_snop_wait_state_dominates_4x4_mfma_finish` | leading LR + SWait | 2 |
| test_dataflow_graph_register_gaps.py | (file-level cleanup) | obsolete `_FIXTURE_GAP_XFAIL_REASON` constant + 9-line stale historic comment block + 5-line stale module docstring paragraph that referenced `xfail(strict=True)` markers no longer present | ~24 |

**Total tests minimized: 44** (36 by 86wq + 8 by 1px6; plus a shared helper that minimizes a further 4 cross-body tests indirectly; plus file-level dead-text cleanup).

### Per-test rationale

All tests in `TestLRSAddrChain`, `TestLWSAddrChain`, `TestGRIncSRDChain`, `TestPackRAW`, `TestDTLm0Tracking`: removed the trailing `make_swait(slot=1, dscnt=0)` + `make_mfma(...)` "realistic GEMM frame" because the property is purely about the producer/consumer pair (e.g. VXor→DSLoad RAW, m0-set→DTL-load RAW, SAdd→BufferLoad SRD WAW, LR→Pack RAW). The trailing instructions never participate in any asserted edge; `compare_graphs` finds the producer/consumer mismatch independently of any timing context.

`TestMFMASelfRAW::test_mfma_acc_chain_reorder`: removed leading LR + SWait. The property pinned is "two MFMAs that share an accumulator must be detected when reordered", which is a `compare_graphs` identity test. The LR + SWait sat before the property and contributed nothing to either reference or subject edge identity.

All `TestMFMAQuadCycleGap` tests: removed the leading LR + SWait. `cumulative_issue_cycles` walks the captured stream from the producer node to the consumer node — instructions before the producer are never part of the gap arithmetic. The `test_mfma_acc_chain_cross_body_uses_unified_simulator` docstring previously walked the explicit "ML-1 MFMA_filler, ML LR, ML SWait, ML MFMA1, NGL MFMA2, NLL MFMA_filler" stream; the docstring was rewritten to reflect the simplified stream.

`test_mfma_pack_to_cvt1_meets_5_cycle_gap_no_failure` and the two `_short_pins_actual_*` siblings: the leading LR + SWait was removed BUT the load-bearing intervening LRs (the ones used to deliberately inflate `cumulative_issue_cycles` to a target value of 3, 4, or 5) were preserved. These intervening LRs sit BETWEEN producer and consumer and are arithmetically critical.

`_build_cross_body_pack_to_cvt_capture` (test helper): removed the leading LR + SWait inside the `ml_prev_instructions` list. The helper builds a cross-body fixture parameterized by `ml_prev_pad_count`, where each pad LR contributes 1 quad-cycle to the cross-body cycle count. The leading LR + SWait sat BEFORE the PackMFMA producer and contributed nothing — the helper's docstring already documented this implicitly ("ML-1 walk starting at PackMFMA").

`TestCumulativeIssueCycles` and `TestSNopWaitState` tests: identical reasoning — leading LR + SWait sit before the producer and play no role in the gap measurement.

`test_cvt_pack_to_mfma_zero_gap_emits_timing_too_close`: only the SWait was removed; the leading LR was preserved because it is the producer of v50..v51 that the CVT consumes. (The LR+SWait pair appears together in many CVT-to-MFMA tests in this file; only the SWait is vestigial because the test only filters for failures with `consumer.category == "MFMA"`, never asserting on the LR→CVT edge.)

Eight remaining CVT tests (sweep completed by 1px6): `test_cvt_pack_to_mfma_meets_2_cycle_gap_no_failure`, `test_cvt_pack_routed_to_quadcycle_not_alu`, `test_cvt_pack_to_mfma_one_short_pins_actual_1`, `test_cvt_pack_to_pack_alu_consumer_takes_alu_exemption`, `test_cvt_to_mfma_cross_body_ml_prev_to_ml_real_cycle_count`, `test_cvt_to_mfma_cross_body_ml_to_ngl_real_cycle_count`, `test_cvt_to_mfma_cross_body_below_threshold_fires_failure`, and `test_cvt_to_mfma_same_body_old_vs_new_formula_documented_divergence`. In every case the SWait sits between the LR producer and the CVT consumer; `cumulative_issue_cycles` walks from the CVT producer forward (never includes instructions before the producer), so the SWait contributes zero to any asserted gap. The failure-filter in each test matches only Pack-producer / MFMA-consumer or Pack-to-Pack failures — no assertion examines the LR→CVT edge. Confirmed by removing each SWait individually and verifying pytest remains 672/2/1. Docstrings in the two cross-body tests that explicitly listed `[LR, SWait, CVT@p_idx]` were updated to `[LR, CVT@p_idx]`.

File-level cleanup: removed the unused `_FIXTURE_GAP_XFAIL_REASON` constant (defined but never referenced — the variable's text described historic xfails that have been removed). Also rewrote the stale module docstring paragraph that said tests "are marked `xfail(strict=True)`" and the stale historic comment block that talked about "remaining xfails in this file" — both wrote about an earlier state of the file and were misleading. The first sentence of each describes the current state: every test in the file passes; this is a regression-pinning file, not a coverage-gap-pinning file.

---

## Part B — xfail / known-not-working itemization

| File:line | Test | Tracked by |
|-----------|------|-----------|
| (none in scope) | — | — |

**Finding**: No tests added since `develop` are marked `@pytest.mark.xfail`, `pytest.skip` (statically), `_FIXTURE_GAP_XFAIL_REASON`, nor documented as "known to not work" / "will fail" / "awaiting fix" / "TODO: fix". The only `xfail`-decorated test in `Tensile/Tests/unit/` is `Tensile/Tests/unit/Common/test_Architectures.py:204` which is pre-existing on `develop` (out of scope).

The two runtime `pytest.skip(...)` calls inside `Tensile/Tests/unit/test_ScheduleCapture.py:907` and `:909` (in `test_mfma_index_increments_and_offsets_by_iteration`) are conditional skips guarding against test-infrastructure feasibility ("Direct MFMAInstruction construction is involved; covered by integration test" / "MFMA construction not feasible in unit test"). They are not "known-not-working" pins — they are "test stub never reaches an assertion because the underlying constructor isn't unit-friendly". The test's body skips before any assertion is ever attempted, regardless of production behavior. These do not represent validator deficiencies; they represent ScheduleCapture testability gaps.

Stale references to `xfail(strict=True)` and "remaining xfails" inside the docstrings/comments of `test_dataflow_graph_register_gaps.py` were updated as part of Part A: the file's xfail markers were dropped along with `_GenericALURule` (wx9.4.4) and the wx9.3 phase-3 operand-slot edge identity landing, but the prose describing them was never refreshed. The single dangling `_FIXTURE_GAP_XFAIL_REASON = (...)` module constant — defined but referenced nowhere — was also removed.

The 1 `xfailed` count in the post-audit pytest summary (matching the baseline) is `Tensile/Tests/unit/Common/test_Architectures.py::Test_isaSupportedSourceFromAsmCaps_8b8e8d40_test_invalid` — pre-existing on `develop`, out of scope.

---

## Tests examined but NOT minimized

- `TestVSwapPair::test_vswap_pair_reorder_detected` and `test_vswap_pair_allocation_invariant`: already minimal (only the two VSwaps required to express the property; `_wrap()` provides the unavoidable filler-MFMA bodies). These were the wx9.3 phase-3 minimization exemplar cited in the bead.
- `TestMFMAQuadCycleGap::test_mfma_acc_chain_cross_body_strict_when_graph_missing`: already minimal — uses the bare cross-body MFMA pair without LR/SWait scaffolding (the test pre-dated my pass with the right shape).
- All `TestPhantomPackScratchReuse` tests in `test_dataflow_graph_phantom_edges.py`: every fake-pack instruction in the fixture is the producer or consumer of a v133 edge that the test counts; the trailing MFMA exists to give v40/v41/v50 graph-node identities (needed for the phantom-edge resolver to surface). All load-bearing.
- All tests in `test_dataflow_graph_lcc.py`: the LCC instructions ARE the property; the producer MFMA + consumer LR are the cycle-walk endpoints. Already minimal.
- All tests in `test_dataflow_graph_barriers.py` (`TestBarrierEdgeFormation`, `TestNoBarrierEdgeWhenWrong`, `TestCrossBodyBarrierEdges`): every LR/SWait/SBarrier/GR in each fixture participates in the asserted barrier-edge formation property. Already minimal.
- `TestCleanComparison::test_identical_graphs_no_failures`, `test_same_dataflow_different_positions_no_failures`, `test_redundant_swaits_and_barriers_in_subject_no_failures`, `test_lcc_included_in_identity_set` in `test_dataflow_graph_comparison.py`: the LR + SWait + MFMA chain IS the dataflow being compared; removing any element changes what `compare_graphs` examines. Already minimal.
- All `TestPerFailureDiagnosis`, `TestRegSetResolution`, `TestRenderStringIdentity`, `TestEdgeIdentityAllocationInvariance`, `TestLegitimateCmsReorderNoFailure`, `TestGenuineOrderInversionStillDetected`, `TestVgprChainReorderDetection`, `TestGRIncReorderDetection`, `TestDiagnoseMissingEdgeDefenses` in `test_dataflow_graph_comparison.py`: each fixture's LR/Pack/MFMA acts as a dataflow producer/consumer for the property under test. Already minimal.
- All tests in `test_validate_lr_before_mfma_graph.py`, `test_validate_pack_graph.py`, `test_validate_gr_not_too_early_graph.py`, `test_dataflow_graph_scc.py`, `test_dataflow_graph_builder.py`, `test_capture_pipeline_checks.py`, `test_failure_formatters.py`, `test_graph_native_validation_base.py`, `test_register_tracing.py`, `test_register.py`, `test_idmap_helper.py`, `test_arch_profile_*`, `test_isa_info_map_no_collect_time_init.py`, `test_LR_Pack_interaction.py`, `test_mfma_reorder_e2e.py`, `test_structural_clone.py`, `test_cms_flag_reconciliation.py`: spot-checked; no recurring vestigial scaffolding pattern observed (each test's instructions correspond to a property expressed by the assertions). The TestCleanComparison-style of `LR + SWait + MFMA chain IS the property` recurs across these files. Not exhaustively audited at this depth — flagged as a future deeper-pass candidate if the audit produces a regression.
- The remaining eight `test_cvt_pack_to_*` and `test_cvt_to_mfma_*` SWait removals (deferred by 86wq) were completed by bead 1px6. All nine CVT-related SWaits have now been removed; the sweep is complete.

---

## Notes for the user (load-bearing surprises, decisions worth reviewing)

- **All CVT-related SWaits removed** (sweep complete as of 1px6): the SWait pattern that appeared between the LR producer and CVT consumer across nine `test_cvt_pack_to_*` / `test_cvt_to_mfma_*` tests has been fully eliminated. The first was removed by 86wq (`test_cvt_pack_to_mfma_zero_gap_emits_timing_too_close`); the remaining eight were removed by 1px6. In every case `cumulative_issue_cycles` walks from the CVT producer forward, so the preceding SWait was never counted in any asserted gap.
- **The historic comment + dead constant `_FIXTURE_GAP_XFAIL_REASON` were a docstring drift, not a code bug**: the constant was defined and never referenced; the comment block above it referenced "remaining xfails" that no longer exist. This is the kind of artifact the audit is designed to catch. The fact that the file's lead docstring still claimed tests are `xfail(strict=True)` while the actual decorators had been dropped reinforces the broader risk that test files in this branch carry stale prose referring to earlier states of the file. A future reviewer should treat docstring claims about pytest marks with suspicion until they re-verify the markers in the test bodies.
- **No tracker beads recommended for filing** — the audit found no xfail/known-not-working tests in scope. If tests later surface that need to be xfailed, they should be filed as their own beads; this audit does not predict any.
- **Some tests carry semantically-rich frames that the audit deliberately preserved**: e.g. `test_mfma_pack_to_cvt1_meets_5_cycle_gap_no_failure` deliberately includes 5 LRs to inflate the gap to exactly 5 quad-cycles — this is the property under test, not scaffolding. The rule of thumb that emerged: instructions that affect `cumulative_issue_cycles` for an asserted edge are load-bearing; instructions whose only effect is to populate the captured stream before/after the asserted producer/consumer pair are vestigial.
