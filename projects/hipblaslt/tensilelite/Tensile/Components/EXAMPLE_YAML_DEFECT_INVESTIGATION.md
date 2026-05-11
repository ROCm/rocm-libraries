# example.yaml OrderInvertedFailure: Root-Cause Investigation

Bead: rocm-libraries-z012.
Investigator: z012-investigator (Claude Opus 4.7, May 2026).
Working tree: `/home/alvasile/rocm-libraries/.worktrees/z012-investigation/`.

---

## 1. Summary

`Tensile/Components/example.yaml` triggers a pre-existing
`OrderInvertedFailure` against `_get_schedule_160x128x64_TF32` at
`KernelWriter.kernelBody`'s graph-comparison assertion. The 8 failures
fire on edges with **state-register resources** — SCC and m0:

- `GRIncA[0] -> GRIncB[1/2]` on `regType=scc` (4 instances across ML-1, ML, NGL bodies)
- `GRA[18] -> GRB[15]` on `regType=m` (3 instances across ML-1, ML bodies)

**Classification: validator defect (b).** Not a schedule defect; not a
config mismatch. The example.yaml's tile config matches the registered
schedule (`MIWaveTile=[5,4]`, `MIWG=[2,2]`, `DepthU=64`, TF32, TN
layout — dispatching through `_get_schedule_160x128x64_TF32`'s `isTN`
branch into `_get_schedule_128x160x64_TF32` after `switch_A_B_schedule`).

**Root cause (one paragraph):** The graph-construction layer
`build_dataflow_graph` collapses physically-distinct instructions sharing
the same canonical render-string into a single `GraphNode`
(`CMSValidator.py:1146` — `nodes_by_identity[node.identity] = node`,
last-writer-wins). Kernel writers routinely emit the SAME canonical
scalar instruction multiple times for state registers — e.g.
`s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter]` is emitted
once for the GRIncA lowering AND once for the GRIncB lowering;
`s_add_u32 m0, m0, 4224` is emitted once per DTL GR group. The
collapsed node's surviving stream-position is unstable across captures
whose scheduling differs (default emits all GRIncBs linearly before all
GRIncAs; CMS interleaves them). When the surviving position differs
between default and subj for the same identity, Phase 1's order check
in `diagnose_missing_edge` surfaces a false-positive `OrderInvertedFailure`
even when the actual emitted ASM has no semantic ordering hazard
(SCC/m0 are clobber-or-not, not order-or-not, and no intervening
clobber exists in subj).

The `§7.3 cross-subiter ALU carve-out was NOT involved`. Both endpoints
of the failing edges are GRInc/GR sub-instructions whose categories
parse to `subiter=0` (GRInc/GR have no trailing-digit subiter
suffix), so the cross-subiter predicate `_node_subiter(p) !=
_node_subiter(c)` evaluates False and the carve-out doesn't fire.

---

## 2. Failure shape (concrete)

Reproduced via direct end-to-end kernel build with
`writer.enable_capture_default_schedule()`:

```
AssertionError: Dataflow graph comparison failed for kernel 160x128x64:
  8 edge difference(s):
  Producer GRIncA[0] @ idx=4 is issued after consumer GRIncB[1] @ idx=0.
  Producer GRIncA[0] @ idx=4 is issued after consumer GRIncB[2] @ idx=1.
  ... (repeated across ML-1, ML, NGL bodies)
  Producer GRA[18] @ idx=116 is issued after consumer GRB[15] @ idx=57.
  ...
```

The label primaries (`GRIncA[0]` / `GRIncB[1]`) are CMS-side
per-category-stream indices. The actual ref-edge endpoints are
SAME-category sub-instruction chains:

| Ref edge resource | Ref producer (default) | Ref consumer (default) |
|---|---|---|
| `regType=scc` | `GRIncB@3.2` `s_cmp_eq_u32 LoopCounterL, StaggerUIter` | `GRIncB@3.3` `s_cselect_b32 s70, WrapUB+0, IncsB+0` |
| `regType=scc` | `GRIncB@3.2` (same) | `GRIncB@3.4` `s_cselect_b32 s71, WrapUB+1, 0` |
| `regType=m`   | `GRB@96.1`  `s_add_u32 m0, m0, 4224` | `GRB@96.2` `buffer_load_dwordx4 ...` |

The `s_cmp_eq_u32 LoopCounterL, StaggerUIter` and
`s_add_u32 m0, m0, 4224` instructions are emitted by the kernel writer
once per A-side GR group AND once per B-side GR group; their canonical
render-strings are byte-identical across the two emissions because
neither carries a side-discriminating operand. The graph collapse
attributes one of the two physical instances to the surviving node;
default and CMS pick different survivors because their stream
interleaving differs.

---

## 3. Why the existing carve-outs don't catch it

- **§7.3 cross-subiter ALU carve-out** (`CMSValidator.py:2517-2519`):
  requires `p_node.subiter(nmps) != c_node.subiter(nmps)`. GRA / GRB /
  GRIncA / GRIncB categories have no trailing-digit subiter and parse
  to subiter=0, so the predicate is always False for these. Doesn't
  fire.
- **SCC OverriddenInputFailure path** (`CMSValidator.py:2549-2579`):
  fires AFTER Phase 1's order check. Phase 1's `OrderInvertedFailure`
  branch triggers first and returns, so the SCC path is unreachable on
  these edges.
- **m0 has no analogous `OverriddenInputFailure` path.** No carve-out
  for m0 exists today.

---

## 4. Fix

`CMSValidator.diagnose_missing_edge` now gates Phase 1's
`OrderInvertedFailure` branch on a state-register identity-collision
detector. The new helper
`_producer_has_same_render_peer_with_distinct_category` scans the
producer's body capture for any other instruction whose
`canonical_str` matches the producer's render but whose category
differs. When the ref edge's `regType` is `scc` or `m` AND such a peer
exists, Phase 1 is skipped — the SCC clobber detector below owns the
real correctness check for SCC, and for m0 the absence of an
intervening m0 writer in `subj_graph` (which would have surfaced as a
distinct missing edge) means no clobber occurred.

The legitimate-reorder tests in `test_dataflow_graph_register_gaps.py
::TestDTLm0Tracking` (`test_dtl_m0_update_before_buffer_load` /
`test_dtl_m0_add_update_before_buffer_load`) construct one m0 writer
plus one BufferLoad per capture; no same-canonical_str peer with a
different category exists in the producer's body, so the gate leaves
those tests on the OrderInverted path unchanged. All 864 unit tests
pass after the fix (baseline at vlt tip `f935e8fd9cf` was 864/2/1; the
"872" figure in the bead description appears to be a mis-recollection).

---

## 5. Files changed

- `projects/hipblaslt/tensilelite/Tensile/Components/CMSValidator.py`
  — added `_producer_has_same_render_peer_with_distinct_category`
  helper above `diagnose_missing_edge`; gated Phase 1 in
  `diagnose_missing_edge` on `is_collapsed_state_edge`; reused
  `_state_reg_kind` for the SCC handling below to avoid the duplicate
  `getattr(ref_resource, "regType", None)` lookup.
- `projects/hipblaslt/tensilelite/Tensile/Components/EXAMPLE_YAML_DEFECT_INVESTIGATION.md`
  — this memo.

No schedule files (`_160x128x64_TF32.py` / `_128x160x64_TF32.py`)
changed. No example.yaml changes. No `ScheduleCapture.py` changes
(an earlier attempt to extend `identity_for` with the category as a
4th tuple element exposed a separate divergence between default-side
and CMS-side Pack categorization that is out of scope for z012; see §6
below).

---

## 6. Out-of-scope findings (potential follow-up beads)

While instrumenting the comparison, the diff between default-side and
CMS-side identity sets surfaced 186 diverging Pack canonical-renders
where default tagged them `PackA0` / `PackB0` (single subiter) but
CMS tagged the same physical instructions `PackA1` / `PackB1`. The
inconsistency is at the tagging layer — default's
`assign_id_to_category_for_default` (`ScheduleCapture.py:1080-1087`)
walks Pack submodules per inner-unroll subiter, while CMS's
`tag_by_origin_id` uses the optSchedule key directly. This divergence
doesn't surface in the standard test suite because identity matching
collapses on canonical_str regardless of category, so the diverging
nodes still compare equal. It would surface immediately if
`identity_for` were extended to discriminate by category. No bead
filed yet — flagging here for awareness.

---

## 7. wlrp-fixture recommendation

example.yaml now validates green end-to-end against
`_get_schedule_160x128x64_TF32` with the carve-out. wlrp's Phase 2
integration tests can switch back to using example.yaml as the canonical
fixture if desired. **Recommendation: leave wlrp on its current
substituted config (16×16×32 TF32, MIWG 2×2, DepthU 32 → MT 128×128)**
because the substituted config exercises the simpler 4×4 single-subiter
PackMFMA path and is faster to build — example.yaml's full 5×4 wave-tile
build is heavier. example.yaml is now suitable for use in any test that
specifically wants to exercise the 5-MIWaveTile / TN-via-swap codepath.

---

## 8. References

- Carve-out: `Tensile/Components/CMSValidator.py:2502-2548`
  (state-register identity-collision gate).
- Helper: `Tensile/Components/CMSValidator.py:2429-2479`
  (`_producer_has_same_render_peer_with_distinct_category`).
- Schedule under test:
  `Tensile/Components/CustomSchedule/gfx950/_160x128x64_TF32.py`
  (TN branch delegates to `_128x160x64_TF32` + `switch_A_B_schedule`).
- Failing input: `Tensile/Components/example.yaml`.
- Related memos:
  - `CROSS_SUBITER_ALU_FP_INVESTIGATION.md` (bwfr — last-writer-wins
    on per-byte vgpr resolver; same family of artifact, different
    layer).
  - `SECTION_7_3_SUPPRESSION_DISCUSSION.md` (uqoz).
  - `SCC_CROSS_BODY_INVESTIGATION.md` (theq — cross-body SCC clearing
    at body boundaries; this fix complements that one by handling the
    same-body identity-collision case).
