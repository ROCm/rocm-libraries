# hdem implementation — Approach A + Approach E

Bead: `rocm-libraries-hdem`. Branch: `hdem-A-plus-E` (forked off
`users/alvasile/validator_long_term_plans` at 40d40a20019).

This memo records what landed for hdem and the verification harness.
Read alongside:

- `ORAM1_PRINCIPLED_APPROACH_INVESTIGATION.md` — the design memo. §2
  (Approach A), §4 (Approach E), §6 (false-positive / false-negative
  budget), §7 (the A+E recommendation).
- `EMISSION_ORDINAL_DESIGN.md` — the 4up4 identity refactor that hdem
  composes on top of.
- `PRELOOP_CAPTURE_PHASE1.md` — the body-label sensitivity context.

## What landed

### Approach A — drop `loop_index` from identity

* `Tensile/Components/ScheduleCapture.py:520-580` —
  `TaggedInstruction.identity_for` now returns
  `(canonical_render, emission_ordinal)` instead of
  `(loop_index, canonical_render, emission_ordinal)`. The
  `body_label` parameter is preserved for backward-compat of callers
  but no longer consulted to compose the tuple. Docstring records the
  Approach A rationale (cross-body pipelining false-positive removal)
  and the per-(body, canonical_render) ordinal-counter retention at
  capture time (cross-body collapse via shared ordinals is the
  desired collapse for the motivating case; residual false-negative
  risk is caught by Approach E at the edge layer).

* `Tensile/Components/ScheduleCapture.py:486-505` —
  `TaggedInstruction.emission_ordinal` docstring updated to describe
  the second-slot position in the new 2-tuple.

* `Tensile/Components/CMSValidator.py:916` — `GraphNode.identity`
  field comment updated.

* `Tensile/Components/CMSValidator.py:3414-3422` — `compare_graphs`
  identity-set gate comment updated (records that the cross-body
  extra-emission gate-layer FN risk is caught at the edge layer per
  Approach E).

### Approach E — body-blind edge-key matching

The literal byte-key tuple proposed in `ORAM1` §4.1 turned out to
lose producer-discrimination in two test scenarios documented below
(swap_pack-vs-LR same-byte and cross-iteration LR_first/LR_second
same-byte). The principled extension actually shipped uses
`producer.identity` and `consumer.identity` as the matching
discriminators — both are body-blind under Approach A and both carry
the rocisa-derived canonical-render text (which encodes operand
register references), so they discriminate "different physical
producer wrote same byte" cases that the bare byte-key proposal
collapsed. xqj3-clean: identity is rocisa-derived. The byte-key
proposal's intended payoff (cross-body matching of the same physical
dataflow) is achieved via identity-collapse on both endpoints rather
than via byte-key collapse; the result is the same body-blindness
without losing producer discrimination.

* `Tensile/Components/CMSValidator.py:1229-1303` —
  `DataflowGraph.edge_keys` rewritten. New tuple shape:
  `(producer.identity, consumer.identity, edge_kind,
  intra_operand_byte_offset, src_operand_slot, sink_operand_slot)`.
  Producer/consumer SchedulePositions (which carried `loop_index`
  and therefore body) are removed from the matching tuple. They
  remain on `DataflowEdge` for diagnostic rendering and the
  same-body Phase 1 order check.

* `Tensile/Components/CMSValidator.py:1102-1110` — `DataflowEdge`
  gains two informational fields (at `:1109-1110`),
  `producer_write_byte_key` and `consumer_read_byte_key`,
  populated at edge-formation time via
  `_byte_keys_for_resource(overlap, name_to_idx=...)` for the
  RAW intra-wave path and via `_resolve_dst_resource(...)` for
  the LDS-reuse barrier path. These are NOT used in the matching
  tuple but are available for diagnostic logging and any future
  consumer that needs them.

* `Tensile/Components/CMSValidator.py:2054-2095` — RAW edge formation
  in `build_dataflow_graph` populates the new byte-key fields using
  the producer's and consumer's body-local `name_to_idx` so
  symbolic-vs-numeric naming asymmetry collapses (same path the
  resolver already uses for latest-writer matching, per
  rocm-libraries-bb34).

* `Tensile/Components/CMSValidator.py:2130-2154` — new
  `_resolve_dst_resource(rocisa_inst)` helper that returns the
  rocisa destination operand of an instruction. Tries `.dst` first
  (DSLoad-shaped), falls back to `getDstParams()[0]`
  (BufferLoad-shaped). Used by the LDS-reuse barrier-edge collector
  to compute byte-keys for both the producer and the consumer
  endpoint.

* `Tensile/Components/CMSValidator.py:2271-2308` — LDS-reuse
  barrier-edge collector populates
  `producer_write_byte_key` from the producer's dst and
  `consumer_read_byte_key` from the consumer's dst (via
  `_resolve_dst_resource`). The two are NOT generally the same
  bytes in LDS-reuse — the producer's dst is the LDS-slot pin on
  the writer side; the consumer's dst is the loaded vgpr range
  on the reader side. (The matching tuple itself uses identities,
  not these byte-keys, so this only matters as informational
  metadata.)

* `Tensile/Components/CMSValidator.py:3610-3615` — `compare_graphs`'s
  `ref_edges_by_key` lookup dict rewritten to the new edge-key
  tuple shape. Multiple ref-edges may share the same edge-key tuple
  (cross-body identity collapse on both endpoints); we pick the
  first such edge as representative for diagnosis.

* `Tensile/Components/CMSValidator.py:3622+` —
  `diagnose_missing_edge` legitimate-reorder branch comment updated
  to reflect the new edge-key shape; Phase 1 order check structure
  unchanged (still consults `p_node.position` and `c_node.position`
  on the subject side, with the same-body guard intact).

## Pre-fix vs post-fix verification

### Test count

Validation baseline (per bead spec, branch tip 40d40a20019):
**1009 passed, 3 skipped, 4 xfailed, 0 failed, 0 xpassed**.

Post-hdem (current branch):
**1018 passed, 3 skipped, 3 xfailed, 0 failed, 0 xpassed**.

Net change:
- +6 new tests in `test_dataflow_graph_hdem.py` (Test #1, #2, #3 per
  bead spec).
- +2 from splitting one test into two:
  `TestEdgeIdentityAllocationInvariance::test_lr_to_mfma_identities_equal_under_allocation_change`
  → `TestEdgeIdentityByteKeyContract::test_lr_to_mfma_identities_equal_under_identical_allocation`
  + `TestEdgeIdentityByteKeyContract::test_lr_to_mfma_identities_differ_under_allocation_change`
  (synthetic allocation-invariance is intentionally dropped under E;
  see the docstring for the rationale and the operative real-build
  property the test now pins).
- +1 from splitting `TestVSwapPair::test_vswap_pair_allocation_invariant`
  similarly into
  `test_vswap_pair_identical_allocation_matches`
  + `test_vswap_pair_renamed_allocation_distinct`.
- +1 from converting the previously-strict-xfail
  `test_whole_kernel_useplrpack_cms_matches_both_defaults` into a
  pass (the exfw-investigated symptom was the pre-hdem body-keyed
  identity collapse; under A+E it disappears).

The xfailed count drops from 4 to 3 because the
`test_whole_kernel_useplrpack_cms_matches_both_defaults` xfail
strict was removed (it now passes). The remaining 3 xfails are
unrelated.

### Pre-fix regression demonstration

The new test
`test_dataflow_graph_hdem.py::TestCrossBodyPipeliningMatches::test_cross_body_pack_pipelining_matches`
demonstrates the pre-fix vs. post-fix difference:

* Pre-fix (verified by stashing the implementation and rerunning):
  ```
  E   Tensile.Components.ScheduleCapture.CaptureConsistencyError:
      compare_graphs: data-flow node identity sets differ.
      in reference but not subject: 1 identities ({'LR': 1});
      first 3: [(-1, 'ds_read_b128 v40, v255 offset:64', 0)];
      in subject but not reference: 1 identities ({'LR': 1});
      first 3: [(1, 'ds_read_b128 v40, v255 offset:64', 0)]
  ```
  The two identities differ only by the leading `loop_index` slot
  (-1 = PRO body, 1 = ML body) — exactly the body-keyed identity
  collapse Approach A removes.

* Post-fix: passes — `compare_graphs` returns `[]` because the
  identity tuple `(canonical_render, emission_ordinal)` is body-blind
  and the edge_keys tuple uses identity-based matching (also
  body-blind).

The companion test
`TestCrossBodyPipeliningMatches::test_cross_body_pack_pipelining_node_collapse`
pins the identity-collapse half directly: the same Pack instruction
in PRO body and in ML body must yield identical identity tuples
post-fix. Pre-fix it failed showing tuples
`(-1, 'ds_read_b128 v40, v255 offset:64', -1)` vs
`(1, 'ds_read_b128 v40, v255 offset:64', -1)`.

### Cross-iteration register rotation pin

`TestCrossIterationRotationStillDistinct::test_two_iterations_register_rotation_match`
constructs two iterations of register-rotated Pack writes (vgpr+0
and vgpr+16) in the same body and pins that REF and SUBJ with
identical such captures yield no failures. The companion
`test_two_iterations_distinct_identities` pins that the rotated
emissions get distinct identities (different canonical_renders).

This is the test that demonstrates Approach D
(canonical_render rotation normalization) is not required: the
rotation case is already handled correctly under A+E because
canonical_render naturally distinguishes rotated emissions, and
both REF and SUBJ have both edges so the set comparison is empty.

### Cross-body extra-write pin

`TestCrossBodyExtraWriteSurfaces::test_extra_pack_producer_with_no_subject_counterpart_surfaces`
constructs a scenario where the reference graph contains a Pack
producer writing v100 + an MFMA reading v100, with no
body-collapsed counterpart in the subject graph (subject's MFMA
reads completely different vgprs). The divergence MUST surface
either as `CaptureConsistencyError` at the identity-set gate (when
the extra producer's identity is genuinely absent from subject) or
as a non-empty Failure list at the edge layer (per ORAM1 §6.1
mitigation argument). The test accepts either outcome and asserts
that the divergence is NOT silently absorbed.

The companion `test_extra_pack_in_pro_with_distinct_consumer_surfaces`
pins the variant where the extra producer lives in PRO body with a
unique consumer in ML body — the consumer's identity in REF is
absent from SUBJ, so the gate fires.

### Identity-shape pinning tests rewritten

Tests previously asserting `len(identity) == 3` or pinning
positional indices `identity[0]` / `identity[1]` / `identity[2]`
were updated:

* `test_dataflow_graph_emission_ordinal.py:148-156` — updated to
  the new 2-tuple shape.
* `test_dataflow_graph_comparison.py:435-465` — updated.
* `test_dataflow_graph_lcc.py:128-134` — updated comment.
* `test_dataflow_graph_builder.py:590-594` — updated comment.

### Tests with semantic updates

* `test_prologue_capture.py::test_preloop_divergence_catches_useplrpack_change`
  — pre-hdem this test asserted that `compare_graphs(g_with,
  g_without)` raised `CaptureConsistencyError` because of the extra
  prologue Pack producers. Under hdem A+E the UsePLRPack-flipped
  difference is exactly the cross-body relocation that A+E is
  designed to make match — both graphs have identical 644 edges and
  182 nodes after the body collapse, and `compare_graphs` returns
  []. The test now pins the body-collapse outcome (graphs
  compare-equal under hdem A+E; node and edge counts agree). The
  cross-body extra-write divergence pin moved to the dedicated
  hdem regression suite (`test_dataflow_graph_hdem.py`).

* `test_prologue_capture.py::test_whole_kernel_useplrpack_cms_matches_both_defaults`
  — the xfail-strict marker (referencing rocm-libraries-exfw) was
  removed because A+E makes the test pass. The exfw symptom (16
  PackA3/PackB3 identities differing by body) was the
  pre-hdem body-keyed identity collapse; under A+E it disappears.

* `test_cross_subiter_alu_carveout_real_kernel.py::test_real_kernel_neutralized_carveout_surfaces_768_pack3_mfma_failures`
  — pre-hdem pinned 768 = 192 × 4 bodies. Under A+E the four
  bodies' identical Pack3 emissions collapse via last-writer-wins
  on `nodes_by_identity` (keyed on
  `(canonical_render, emission_ordinal)` with `loop_index` dropped)
  to a single node entry per shape. The test now pins 192 (96
  PackA3 + 96 PackB3) — one body's worth — and the
  `body_counts == {"ML": 192, ...}` assertion was removed because
  body discrimination has dropped from the failure stream.

* `test_dataflow_graph_register_gaps.py::TestVSwapPair::test_vswap_pair_allocation_invariant`
  — split into `_identical_allocation_matches` (pin: yes, matches)
  and `_renamed_allocation_distinct` (pin: yes, differs). The
  synthetic allocation-invariance contract is intentionally dropped
  under E because canonical_render encodes operand register text;
  in the real motivating case both captures share an allocator
  snapshot so the divergence does not occur.

## References

- `ORAM1_PRINCIPLED_APPROACH_INVESTIGATION.md` — the design memo;
  the recommendation summary at §7 / §12 is what hdem implements
  (§4's literal byte-key proposal was extended to identity-based
  matching for the reasons documented above).
- `EMISSION_ORDINAL_DESIGN.md` — the 4up4 identity refactor that
  hdem composes on top of.
- `PRELOOP_CAPTURE_PHASE1.md` — body-label sensitivity context.

## Implementation surprises

1. **The literal byte-key proposal in ORAM1 §4 lost
   producer-discrimination.** The `swap_pack` test
   (`test_validate_pack_graph.py::TestSwapPackGraph::test_pack_before_swap_orderinverted`)
   and the cross-iteration LR_first/LR_second test
   (`test_validate_gr_not_too_early_graph.py::TestGRNotTooEarlyDtlPlusLdsBufGraph::test_negative_one_prev_iter_lr0_not_drained`)
   both fail under a pure byte-key tuple because two distinct
   producers writing the same physical byte collapse on the
   producer side. The principled extension is identity-based
   matching: identity is body-blind under A AND
   producer-discriminative because `canonical_render` encodes
   operand register references and `emission_ordinal` discriminates
   per-render-text emission counters within the same body. The
   ORAM1 memo's design rationale ("the same physical byte flowed
   from one writer to one reader") is preserved at the edge layer;
   it just consults identity rather than raw byte-keys.

2. **The synthetic allocation-invariance tests pinned a contract
   that is incompatible with both byte-key matching AND
   identity-based matching.** Both proposals embed
   physical-register information into the matching tuple (byte-keys
   directly; identity via canonical_render). The synthetic test
   case (different physical registers, same logical structure)
   doesn't occur in the real motivating case (CMS-vs-default within
   the same `kernelBody` invocation always shares the allocator
   snapshot). The tests were updated to pin the operative property
   (allocation-EQUAL captures match; allocation-DIFFERENT captures
   produce distinct keys).

3. **Line-number drift from the design memo.** The memo's pre-4up4
   citations had `identity_for` at lines 507-536, `edge_keys` at
   1238-1241, `ref_edges_by_key` at 3349-3354, Phase 1 order check
   at 3434-3457. After 4up4 (`83c5d507b49`) and the additional
   docstring expansions in this implementation, the corresponding
   post-fix locations are:
   - `identity_for` → `ScheduleCapture.py:522-585`
   - `edge_keys` → `CMSValidator.py:1229-1303`
   - `ref_edges_by_key` → `CMSValidator.py:3610-3615`
   - Phase 1 order check → `CMSValidator.py:3694-3735`
