# C3g implementation plan — `cumulative_issue_cycles` + remaining `body_label` control-flow consumers

**Bead:** `rocm-libraries-ktwt`
**Depends on:** `rocm-libraries-i190` (C3f, must be closed)
**Blocks:** `rocm-libraries-si5f` (C3h)
**Commit alias:** C3g

---

## §1 Scope

Two distinct migration targets share this commit:

1. **`cumulative_issue_cycles`**: replace the body-discovery scaffolding (three clusters of `_BODY_BUILD_ORDER` enumeration + per-body instruction walks) with a single flat walk over the `sorted_nodes` sequence that `build_dataflow_graph` already produces on the graph. Core arithmetic (MFMA contention, type-switch +1 stall, per-instruction issue accumulation) is body-agnostic and unchanged.

2. **Remaining control-flow `body_label` consumers** (per `UNROLLED_VALIDATION_ANSWERS.md §Q11`): five sites outside `cumulative_issue_cycles` still gate control flow on body membership or `_BODY_BUILD_ORDER` enumeration. Migrate each. Diagnostic-annotation sites (body_label field, `body_for()`, `cms_node_label`, failure renderers — ~45 sites) are untouched.

Validator state after this commit: unchanged from C3f (still RED from C1 until C3h removes xfail).

---

## §2 Investigation findings

### A. `cumulative_issue_cycles` current shape (actual line ranges after C3a–C3f drift)

The function lives at **line 2506–2704**. Body-discovery scaffolding:

| Line range | What it does | Why it exists today |
|---|---|---|
| 2549–2551 | `captures = getattr(graph, "captures", None)` + early return | Defensive: old path needed `captures` dict to pull per-body instruction lists |
| 2567 | `if not (producer.position < consumer.position): return 0` | Guards same-body ordering using `SchedulePosition`; correct, but expressed as "position" not "unrolled_position" |
| 2580–2588 | `for i, label in enumerate(_BODY_BUILD_ORDER): if label == producer.body_label: p_body_idx = i` | Maps producer/consumer body labels to traversal start/end indices |
| 2590–2604 | Fallback slot-key helpers for `ti is p_ti` identity matching | Needed because old body-per-body walk had to search by identity within each body's instruction list |
| 2616–2700 | `for body_i in range(p_body_idx, c_body_idx + 1)` outer loop + inner `for i, ti in enumerate(instructions)` per-body search | Traverses bodies in `_BODY_BUILD_ORDER` order; locates producer/consumer by `ti is p_ti` identity matching within each body's raw `instructions` list |
| 2646 | `if label == consumer.body_label:` inner gate | Determines where to stop in the consumer's body |
| 2687 | `if p_issue_start is None and i == start_idx and label == producer.body_label:` | Records producer's issue-cycle start |

**The key observation:** the function currently walks `body.instructions` (raw `TaggedInstruction` list from the `LoopBodyCapture`), not the `GraphNode` sequence in `sorted_nodes`. The `sorted_nodes` list built by `build_dataflow_graph` Phase 2 is already ordered by `unrolled_position` and spans all bodies (PRO → ML_PREV → ML → NGL → NLL). However, `sorted_nodes` contains only "dataflow" instructions (those in `dataflow_ti_ids`), not scheduler-choice SWait/SBarrier/SNop. The raw `body.instructions` walk is needed for per-instruction issue accumulation because SWait/SNop/SBarrier contribute issue cycles even though they are not GraphNodes.

**Resolution:** keep walking the raw instruction stream, but source it from the graph's pre-assembled unrolled walk rather than reconstructing it from body labels. The graph needs to expose a flat unrolled instruction stream (not just the node sequence).

### B. How the unrolled instruction stream is already available

`build_dataflow_graph` Phase 1 populates `nodes_per_body[label]` for GraphNodes and attaches `_graph_nodes` to each `LoopBodyCapture`. But the raw `body.instructions` (`LoopBodyCapture.instructions`) lists are already on each capture. The `DataflowGraph` carries `graph.captures` — the same dict `cumulative_issue_cycles` already reads via `getattr(graph, "captures", None)`.

The migration does NOT need to change the storage layout. Instead:

1. The body-discovery phase (which body idx is the producer in? which is the consumer in?) is replaced by reading `producer.unrolled_position` and `consumer.unrolled_position` directly — these are already stamped on every GraphNode by `build_dataflow_graph` Phase 1.

2. The traversal loop replaces the `_BODY_BUILD_ORDER` enumeration + per-body instruction walk with:
   - Walk bodies in `_BODY_BUILD_ORDER` order (unchanged); skip bodies before the producer's body and after the consumer's body using body-label comparison against producer/consumer node's `body_label`.
   - Within each body, the start/end instruction index is located by matching `unrolled_position` of the producer and consumer against a per-body running counter (not `ti is p_ti` identity matching).

3. Alternatively — cleaner — the simulator can accumulate a global instruction counter alongside the body walk, and the producer/consumer positions reduce to: "start issuing when global counter reaches producer's `unrolled_position`; stop when it reaches consumer's `unrolled_position`."

**Chosen approach (simplest, no new storage):** keep the outer `_BODY_BUILD_ORDER` loop; replace the inner per-body `ti is p_ti` identity search with: iterate all instructions in the body using a running `global_ti_count`, compare `global_ti_count` against `producer.unrolled_position` and `consumer.unrolled_position` to determine start and stop. The `unrolled_position` is already the global flat index assigned by `assign_stream_indices_for_body` calls during Phase 1.

**Body-loop verdict (principled, not tactical):** The outer `_BODY_BUILD_ORDER` loop is kept because `body.instructions` (the raw TaggedInstruction list including SWaits/SNops not in the graph nodes) is the correct source of issue-cycle data. The graph's `sorted_nodes` is insufficient for this walk — it excludes those instructions. There is no flat unrolled raw-instruction stream stored on the graph; building one would require new storage. Keeping the per-body outer loop with a global position counter is the principled approach given the existing data model. An alternative iterating `subj_graph.nodes` directly would be WRONG for the same reason the `all_nodes_in_order` simplification is wrong (§2F). The design is consistent.

Wait — `unrolled_position` is assigned per GraphNode, not per raw TaggedInstruction. SWaits/SNops are not in `nodes_list`. Check how `unrolled_position` is derived.

Looking at the code: `unrolled_pos` is incremented for every instruction in the body's raw instruction list (line ~2050 area), not just dataflow ones. Each TaggedInstruction in `body.instructions` gets a sequential position. The GraphNode's `unrolled_position` matches the `unrolled_pos` value from when its `tagged_inst` was encountered.

Therefore: iterating `body.instructions` with a running counter that matches the `unrolled_position` stamping in Phase 1 directly identifies producer and consumer by position value. The `ti is p_ti` search becomes: "continue iterating until `running_pos == producer.unrolled_position`."

### C. The `captures` early-return

The defensive `if not captures: return 0` at line 2549 can be kept; it fires only on ill-formed test stubs that don't set up captures at all. All well-formed graphs have `captures`. No change needed here.

### D. The `position < consumer.position` ordering guard (line 2567)

Currently uses `SchedulePosition.__lt__` on `producer.position`. Since `unrolled_position` is a strictly monotone int, this simplifies to `producer.unrolled_position < consumer.unrolled_position`. The comparison is semantically equivalent (same ordering guarantee). Migrate in place.

### E. Q11 control-flow consumers outside `cumulative_issue_cycles`

From `UNROLLED_VALIDATION_ANSWERS.md §Q11`, the non-`cumulative_issue_cycles` control-flow sites:

| File:line (post-drift, verify at implementation time) | Site | Current behavior | Migration |
|---|---|---|---|
| ~1360 | `all_nodes_in_order` property: `for label in _BODY_BUILD_ORDER: cap = self.captures.get(label); for node in cap._graph_nodes:` | Iterates bodies in build order, yields per-body nodes | Replace: yield from `self._sorted_nodes` (the `sorted_nodes` list stashed on the graph at `DataflowGraph.__init__` time — already exists as `graph.nodes` which IS the sorted list post-C3b). If `graph.nodes` is the position-ordered sequence, `all_nodes_in_order` simply yields from it. |
| ~2088–2091 | Stash `_graph_nodes` on each `LoopBodyCapture`: `for label in _BODY_BUILD_ORDER: body_cap._graph_nodes = nodes_per_body[label]` | Per-body sidecar for `body_for()` and diagnostic helpers | Keep: `_graph_nodes` is needed by `body_for()` and diagnostic consumers. The loop itself is fine — it's wiring, not control-flow gating. Classification: KEEP (not a control-flow consumer of body_label in the dataflow sense). |
| ~2155–2167 | `prev_body_label` SCC-clearing at body boundary in Phase 2 edge walk | Detects body transitions to clear SCC byte_keys from `latest_writer` | Keep logic; the transition detection (`node.body_label != prev_body_label`) is correct and necessary (SCC does not survive across body boundaries). The condition generalizes naturally: under the unrolled walk, body boundary transitions are the same as today because two ML iter copies have DISTINCT body_labels (`BODY_LABEL_ML_PREV` vs `BODY_LABEL_ML`). No change needed. Classification: KEEP (semantically correct as-is). |
| ~2277–2279 | `all_nodes_in_order = []; for label in _BODY_BUILD_ORDER: all_nodes_in_order.extend(nodes_per_body[label])` then passed to `_collect_barrier_edges` | Assembles flat node list for barrier edge collection | Replace: `all_nodes_in_order = sorted_nodes` (already computed). `sorted_nodes` is the unrolled-position-ordered flat list of all dataflow nodes. Same shape, no extra work. |
| ~4282 | `if (p_node.body_label != c_node.body_label and waits and subj_graph.any_drains(waits, p_node)):` | Suppresses cross-body wait-coverage classification | **Wait-gate semantics audit required.** The gate suppresses `UnexplainedMissingEdgeError` for loop-carried handoffs where a wait drains the producer across bodies. Under Resolution 1 (C3c), ML iter-0 uses `BODY_LABEL_ML_PREV` and iter-1 uses `BODY_LABEL_ML` — so the existing `p_node.body_label != c_node.body_label` check ALREADY fires for cross-iter ML edges (they have distinct body_labels). The proposed migration to `p_node.iter_index != c_node.iter_index or p_node.body_label != c_node.body_label` is therefore redundant for the ML cross-iter case (iter_index differs AND body_label already differs). The addition of the `iter_index` disjunct only has effect if two nodes with the same body_label can have different iter_index values — which cannot happen under the current single-label-per-iter assignment. **Verdict: the proposed migration is a no-op in practice under Resolution 1 and can be kept for defensive clarity, but the comment in §2E claiming it covers "same-body cross-iter suppression" is incorrect — no two nodes share a body_label across iter copies.** The gate should either stay as-is (`body_label != body_label`) or be rewritten to the cleaner `p_node.unrolled_position < c_node.unrolled_position` (which is already guaranteed true for any edge reaching this point), making the gate unconditionally fire when waits drain the producer. The `unrolled_position`-only form is the principled rewrite; the `iter_index or body_label` form is redundant but harmless. |

### F. `all_nodes_in_order` — post-C3b actual shape

Post-C3b/C3c, `DataflowGraph.nodes` IS the `sorted_nodes` list (a Python list of `GraphNode` in `unrolled_position` order). The `all_nodes_in_order` property at line 1351–1365 still uses the per-body `_BODY_BUILD_ORDER` + `cap._graph_nodes` walk. This is a redundant implementation of the same ordering — `_graph_nodes` per body, concatenated in `_BODY_BUILD_ORDER` order, gives the same sequence as `sorted_nodes` because that's how `sorted_nodes` was built.

**BLOCKER: `yield from self.nodes` is WRONG here.** `self.nodes` (= `sorted_nodes`) is built from `nodes_list`, which only includes nodes that pass the `data_flow_instructions` filter — SWait, SBarrier, SNop, and SSetPrior nodes are excluded from `nodes_list` by `_NO_DATAFLOW_IDENTITY_CATEGORIES`. However, `nodes_per_body[label]` (which populates `cap._graph_nodes`) receives EVERY node produced by `_make_node`, including SWait and SBarrier nodes. The three queue-simulation helpers that call `all_nodes_in_order` — `queue_depth_at`, `wait_drains_producer`, and any_drains` (transitively) — all filter for `n.category == SWAIT_CATEGORY` to simulate queue drain events. Switching to `yield from self.nodes` would silently drop all SWait nodes from these walks, causing every queue-depth calculation to return incorrect (inflated) values and every `any_drains` check to return False, which would suppress MissingWaitFailure emission.

**Correct migration:** `all_nodes_in_order` MUST continue to yield all nodes including SWait/SBarrier. The current `cap._graph_nodes` walk already delivers this. The simplification for this property is limited to: keep the existing body loop but skip the `hasattr(cap, '_graph_nodes')` guard once the stash loop at line 2088 is guaranteed to run. Do NOT switch to `yield from self.nodes`. The `_BODY_BUILD_ORDER` outer loop in `all_nodes_in_order` is correct and must be kept.

### G. Test re-pinning requirement (Q12 confirmation)

All tests in `TestCumulativeIssueCycles` and cross-body timing tests use synthetic captures with one body instance each. The unrolled walk produces identical cycle counts for single-instance bodies (the producer and consumer are in the same traversal order, the global counter increments identically). Zero re-pinning required.

**Q12 empirical status:** this claim is reasoned from fixture structure, not verified by a test run. The reasoning is sound (synthetic fixtures have no ML iter multiplicity), but must be confirmed by diffing test output before/after at step 7. Any divergence is a plan defect.

One new test is needed: `test_cross_iter_ml_cycle_count` — verify that cycle gap between iter-0 producer and iter-1 consumer includes all intervening instructions across the ML_PREV/ML boundary.

---

## §3 Design — new `cumulative_issue_cycles` iteration shape

Replace the two-phase (body-discovery + per-body walk) with a **single-pass global walk** over body instruction lists, using `unrolled_position` as the locator:

```
Algorithm:
  global_pos = 0
  current_issue = 0
  mfma_free_at = 0
  last_mfma_class = None
  last_mfma_issue = -1
  p_issue_start = None
  c_issue_start = None
  found_consumer = False

  for each body label in _BODY_BUILD_ORDER:
    body = captures.get(label)
    if body is None: continue
    for each ti in body.instructions:
      # Determine whether this ti is the producer or consumer
      at_producer = (global_pos == producer.unrolled_position)
      at_consumer = (global_pos == consumer.unrolled_position)
      # ... accumulate issue state (unchanged MFMA arithmetic) ...
      if at_producer: p_issue_start = current_issue
      if at_consumer: c_issue_start = current_issue; break (both loops)
      current_issue += profile.min_issue_quad_cycles_for(inst)
      global_pos += 1
    else:
      global_pos += 1  (handled inside loop increment)
    if found_consumer: break

  return c_issue_start - p_issue_start - 1  (unchanged)
```

**Key properties:**
- Outer `_BODY_BUILD_ORDER` loop is kept (correct traversal order; no new storage needed).
- Inner body walk uses `global_pos` incremented per raw TaggedInstruction — matches `unrolled_position` stamping from Phase 1 exactly.
- `p_body_idx` / `c_body_idx` discovery is deleted; replaced by `global_pos == producer.unrolled_position`.
- The `ti is p_ti` slot-key fallback is deleted; position match is unambiguous.
- The `start_idx` skip-to-producer inner loop is deleted; the global counter accumulates from 0 and the producer detection fires at the right position.
- Simulator state (MFMA contention, type-switch) still persists across body boundaries exactly as today — the flattened loop naturally carries it.
- Early exit: once consumer is found, break both loops via a flag.

**Guard:** `if producer.unrolled_position >= consumer.unrolled_position: return 0` replaces the `SchedulePosition.__lt__` guard. Same semantics, cleaner expression.

---

## §4 Call-site migration table for `body_label` control-flow consumers

| Site (verify line at implementation time) | Current pattern | Migration | Test impact |
|---|---|---|---|
| `cumulative_issue_cycles` body-discovery (lines 2580–2588) | `for i, label in enumerate(_BODY_BUILD_ORDER): if label == producer.body_label: p_body_idx = i` | Delete; use `producer.unrolled_position` directly | None (zero re-pinning) |
| `cumulative_issue_cycles` outer traversal (lines 2616–2658) | `for body_i in range(p_body_idx, c_body_idx + 1): label = _BODY_BUILD_ORDER[body_i]; body = captures.get(label)` | Replace with single-pass global walk (§3) | None |
| `cumulative_issue_cycles` consumer-body gate (line 2646) | `if label == consumer.body_label:` | Delete; consumer detected by `global_pos == consumer.unrolled_position` | None |
| `cumulative_issue_cycles` producer start recording (line 2687) | `if p_issue_start is None and i == start_idx and label == producer.body_label:` | Replace with `if at_producer:` (per §3) | None |
| `all_nodes_in_order` property (~line 1360) | `for label in _BODY_BUILD_ORDER: ... for node in cap._graph_nodes: yield node` | **DO NOT replace with `yield from self.nodes` — see §2F blocker.** Keep the existing `_BODY_BUILD_ORDER`+`cap._graph_nodes` walk; it is correct and includes SWait/SBarrier nodes that the queue helpers require. | No change needed |
| `all_nodes_in_order` local in `build_dataflow_graph` (~line 2277) | `for label in _BODY_BUILD_ORDER: all_nodes_in_order.extend(nodes_per_body[label])` | Replace with `all_nodes_in_order = sorted_nodes` | None |
| `diagnose_missing_edge` Phase 2 cross-body wait gate (~line 4282) | `if p_node.body_label != c_node.body_label and waits and ...` | **Preferred principled rewrite:** `if waits and subj_graph.any_drains(waits, p_node):` — drop the body_label guard entirely. The gate is inside the `if not failures:` block, which is only reached when no MissingWaitFailure was emitted. Any edge here with draining waits is a loop-carried handoff regardless of body relationship. If the `iter_index or body_label` form is kept instead, verify this is not a no-op under Resolution 1 (see §2E audit above). | Verify existing cross-body wait-suppression tests pass unchanged |

**Sites explicitly classified as KEEP (no migration):**

- `~2088–2091`: stash `_graph_nodes` on captures — wiring, not dataflow gating. Keep.
- `~2155–2167`: SCC-clearing `prev_body_label` in Phase 2 — semantically correct under unrolled walk (ML_PREV and ML have distinct body_labels; transition fires naturally). Keep.
- All `~45` diagnostic-annotation sites — kept unchanged per Q11 classification.

---

## §5 Step order

1. Verify `C3f (rocm-libraries-i190)` is closed. Do not start C3g work until it is.

2. Read current line numbers for each site in the migration table (they will have drifted from the plan's reference lines). Confirm the Q11 classification for any sites not listed above.

3. Migrate `cumulative_issue_cycles`:
   a. Delete the `p_body_idx / c_body_idx` discovery block.
   b. Delete the slot-key fallback helpers (`_slot_key`, `p_key`, `c_key`).
   c. Replace the outer `range(p_body_idx, c_body_idx+1)` loop with the single-pass global walk per §3.
   d. Replace the `SchedulePosition.__lt__` guard with `unrolled_position` comparison.

4. **Do not migrate `all_nodes_in_order` property** (see §2F blocker: `self.nodes` excludes SWait/SBarrier nodes that queue helpers need). Leave it as-is.

5. Replace the `all_nodes_in_order` local list in `build_dataflow_graph` (~line 2277) with `all_nodes_in_order = sorted_nodes`.

6. Migrate the `diagnose_missing_edge` cross-body wait gate (~line 4282) to include iter-boundary check.

7. Run the full unit suite:
   ```bash
   PYTHONPATH=$PWD /home/alvasile/venv/bin/python3 -m pytest \
     --ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py \
     -v --tb=short Tensile/Tests/unit/ 2>&1 | tee /tmp/post_c3g.txt
   ```
   Diff against C3f baseline. Only expected changes (if any) may appear.

8. Add `test_cross_iter_ml_cycle_count` in `test_dataflow_graph_register_gaps.py`: construct a two-body synthetic graph (ML_PREV + ML each with a simple instruction chain), produce GraphNodes with correct `unrolled_position` values, call `cumulative_issue_cycles`, pin the result.

---

## §6 Validation — expected delta from C3f

- Zero new test failures beyond those already RED from C1.
- Zero re-pinned test assertions (Q12 confirmation).
- `TestCumulativeIssueCycles` passes unchanged.
- All cross-body timing tests (`test_mfma_acc_chain_cross_body_*`, `test_cvt_to_mfma_cross_body_*`, `test_mfma_pack_to_cvt1_cross_body_*`) pass unchanged.
- `queue_depth_at` and `producer_queue_position` return identical values — `all_nodes_in_order` is unchanged (SWait/SBarrier nodes still present via `cap._graph_nodes` walk).
- New test `test_cross_iter_ml_cycle_count` passes (new, not a re-pin).

---

## §7 Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| `unrolled_position` counter in the new walk diverges from Phase 1 stamping | Low — Phase 1 iterates the same `body.instructions` lists in the same body order and increments `unrolled_pos` per instruction | Unit test the new walk on a synthetic fixture with known `unrolled_position` assignments; verify producer and consumer are detected at the right positions |
| `all_nodes_in_order` diverges after switch to `yield from self.nodes` | **ELIMINATED** — `yield from self.nodes` migration is cancelled (§2F). `self.nodes` excludes SWait/SBarrier; `all_nodes_in_order` MUST keep the `cap._graph_nodes` walk. No risk of divergence since no change is made. | N/A — `all_nodes_in_order` is not modified |
| The `diagnose_missing_edge` wait-gate change broadens suppression incorrectly | Low — adding `iter_index != iter_index` only enables suppression for cross-iter edges that already have wait coverage; this is the correct behavior | Verify the existing `test_wait_coverage_suppression_*` tests pass; if none exist, add one |
| `captures` early-return in `cumulative_issue_cycles` masks graph errors on production kernels | Negligible — production graphs always have `captures`; defensive guard stays | Keep the guard; no change |

---

## §8 New beads

No new beads required for C3g itself. If the `diagnose_missing_edge` wait-gate migration surfaces unexpected cross-iter edges being suppressed, file a P0 bead with `br dep add si5f <new-bead>` before proceeding to C3h.

The SHADOW capture infrastructure defect (`_build_shadow_cms_pair` returns `None`) referenced in `UNROLLED_VALIDATION_ANSWERS.md §Beads filed` must be resolved before C3h acceptance (not a blocker for C3g itself — C3g does not run n7og fixtures). Ensure that bead is in the tracker and blocked against `rocm-libraries-si5f`.
