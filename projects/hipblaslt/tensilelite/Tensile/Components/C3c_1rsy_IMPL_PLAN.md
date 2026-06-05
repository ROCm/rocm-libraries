# C3c Implementation Plan — `build_dataflow_graph` rewrite consuming `UnrolledCapture`

**Bead:** `rocm-libraries-1rsy`
**Status:** Planning (read-only investigation)
**Date:** 2026-06-05

---

## §1 Scope

C3c rewrites `build_dataflow_graph` in `CMSValidator.py` to call `UnrolledCapture.from_four_part_capture(fpc)` (already landed in C3a/abgv) and walk the resulting unrolled stream. Phase 1 constructs one `GraphNode` per unrolled-stream position — ML iter copies get distinct `GraphNode` objects sharing the same `identity` (iter-blind). Phase 2 does a single `latest_writer` walk over the full unrolled sequence, with SCC-only clearing at body boundaries and no other per-body resets. `DataflowEdge` gains six diagnostic annotation fields (not in `edge_keys()`). The test fixture at `test_dataflow_graph_builder.py:519-523` is re-pinned because ML now contributes `ML_MAT_COUNT` nodes per instruction. Validator state remains partly RED after C3c — `edge_keys()` and `compare_graphs` stay on the identity-tuple basis until C3d/C3e.

**USER-RESOLVED 2026-06-05:** Both blockers have explicit user direction. Original gap text preserved for context; resolutions follow.

**Resolution 1 — ML-1 body (rocm-libraries-tom4): MERGED INTO UnrolledCapture as ML[0].** ML-1 is renamed to ML[0] in the unrolled timeline; the existing `main_loop_prev` stream is materialized once as `UnrolledIterRecord(body_label=BODY_LABEL_ML_PREV, iter_index=0, ...)`. The existing `main_loop` stream is materialized once as `UnrolledIterRecord(body_label=BODY_LABEL_ML, iter_index=1, ...)`. ML_MAT_COUNT stays = 2; each iter copy uses a **different** FourPartCapture field — NOT the same field twice. The unrolled timeline becomes: PRO → ML[0]=ML-1's instructions → ML[1]=ML's instructions → NGL → NLL. Identity iter-blindness still holds: instructions that appear in both ML-1 and ML with the same canonical_render produce identical identities; pipelining cancels in set-diff. `UnrolledCapture.from_four_part_capture` is extended in C3c to include the ML-1 record. The C3a-era documentation of "ML appears ML_MAT_COUNT times reusing the SAME underlying TaggedInstruction objects" is REVISED: each iter copy uses its own FourPartCapture field. The shared-reference invariant from C3a applies WITHIN an iter copy, not ACROSS iter copies.

**Resolution 2 — Re-fixture scope (rocm-libraries-2k8w): ENUMERATE UPFRONT, RE-FIXTURE IN THIS COMMIT.** Per `feedback_no_deferred_discoveries`. The implementation step list must include an enumeration pass: grep every `len(.*\.nodes` / `len(.*\.edges` / category-count assertion under `Tensile/Tests/`, classify each as "ML-body affected" vs "not affected" by checking whether the asserted count comes from instructions in main_loop or main_loop_prev fixtures, re-fixture every affected assertion in the C3c commit. The acceptance criteria gains: "no skipped/xfailed test masks an ML-body count assertion that would now break." Tests known a priori: `test_dataflow_graph_builder.py:519-523`, `:144`, `:589` (the verifier-named ones); the full set comes from the enumeration pass.

---

## §2 Investigation Findings

### A — Current `build_dataflow_graph` shape

**File:line:** `CMSValidator.py:1875` (function definition).

**Input:** A `FourPartCapture` (or `None` → returns empty graph).

**Phase 1 (node construction):** Iterates `_BODY_BUILD_ORDER = (PRO, ML-1, ML, NGL, NLL)`. For each present body, calls `assign_stream_indices_for_body(body.instructions)` to collapse `(mfma_index, sequence)` slot-lex order into a per-body monotonic `stream_index`, then `data_flow_instructions(body)` to filter out SWait/SBarrier/SNop/SSetPrio, then `_make_node(tagged_inst, label, stream_index, arch_profile)` for each instruction. Resulting nodes are appended to `nodes_list` (dataflow-participating) and `nodes_per_body[label]` (all, including sync ops). The sidecar `body._graph_nodes = nodes_per_body[label]` is written at the end of each body.

**Phase 2 (latest_writer walk):** Sorts `nodes_list` by `SchedulePosition` (which encodes `loop_index` from `BODY_LABEL_TO_LOOP_INDEX`, then `stream_index`). Initializes `latest_writer = {}` once before the loop. Walks in sorted order, maintaining a `prev_body_label` to detect body-boundary transitions. At each transition: clears SCC entries from `latest_writer` (keys where `bk[0] == "scc"`). Non-SCC entries are preserved across body boundaries — this is already the correct cross-body behavior for VGPR/SGPR/memory dataflow. For each node: reads first (consult `latest_writer`, emit `DataflowEdge`s via `_resolve_producers`), then writes (`latest_writer[bk] = (node, write_resource, write_slot)` and `byte_key_writers.setdefault(bk, []).append((node, node.position))`).

**Per-body resets that need to GO AWAY in C3c:** There are no per-body resets of `latest_writer` in the current code (non-SCC entries are already preserved). The existing code already carries `latest_writer` across body boundaries. The only reset is SCC-clearing at body boundaries — that is preserved. VERIFIED: `CMSValidator.py:2076-2106` confirms `latest_writer = {}` is initialized once before the walk, and the only per-body action is the SCC-key filter at body-label transitions. Decision 1 is CORRECT.

**ML-1 body status in the unrolled stream (USER-RESOLVED — see Resolution 1 above):** ML-1 is merged INTO `UnrolledCapture` as ML[0] (iter_index=0). The existing `main_loop_prev` stream becomes the first ML iter copy; the existing `main_loop` stream becomes the second (iter_index=1). C3c MUST extend `UnrolledCapture.from_four_part_capture` accordingly. ML_MAT_COUNT stays = 2. The two iter copies use **different** FourPartCapture fields, NOT the same field twice.

**Key discovery:** The current Phase 2 is ALREADY cross-body for non-SCC resources. The per-body isolation problem that causes the 192 NLL extras is not a `latest_writer` reset at body boundaries — it is that the current Phase 1 produces at most ONE GraphNode per ML instruction (no iter copies), and the walk processes ML once. When CMS places ML_iter[K+1]'s MFMA consumer before ML_iter[K]'s PackB3 writer in NLL's stream, the writer was never seen before the consumer in the unified walk. The fix is materializing ML twice (two iter copies) in the unrolled stream, so ML_iter[0]'s writes ARE visible when ML_iter[1]'s reads are resolved.

### B — DataflowEdge current shape

`DataflowEdge` (defined at `CMSValidator.py:1038`) carries:
- `producer: GraphNode`
- `consumer: GraphNode`
- `resource: object` (RegisterContainer | MemoryRegion)
- `edge_kind: str`
- `intra_operand_byte_offset: tuple = ()`
- `src_operand_slot: int = 0`
- `sink_operand_slot: int = 0`
- `producer_write_byte_key: tuple = ()`
- `consumer_read_byte_key: tuple = ()`

C3c adds six diagnostic annotation fields:
- `producer_iter_index: int = 0`
- `consumer_iter_index: int = 0`
- `producer_body_label: str = ""`
- `consumer_body_label: str = ""`
- `producer_unrolled_position: int = -1`
- `consumer_unrolled_position: int = -1`

`edge_keys()` currently returns `{(e.producer.identity, e.consumer.identity, e.edge_kind, e.intra_operand_byte_offset, e.src_operand_slot, e.sink_operand_slot) for e in self.edges}`. The six new annotation fields do NOT appear in `edge_keys()` — that migration is C3d. Confirmed: the current `edge_keys()` does not include `producer_write_byte_key`, `consumer_read_byte_key`, or any body/iter fields.

### C — GraphNode shape

`GraphNode` (defined at `CMSValidator.py:897`) carries:
- `identity: tuple`
- `position: SchedulePosition`
- `category: str`
- `rocisa_inst: object`
- `tagged_inst: TaggedInstruction`
- `body_label: str`
- `name: str = ""`
- `issue_cycles: int = 1`

C3c needs to add two new fields to carry unrolled-walk context:
- `unrolled_position: int = -1` — the global position in the unrolled stream (0-based). Default -1 means "not yet assigned" (backwards-compatible sentinel for nodes from the old walk).
- `iter_index: int = 0` — the ML iter copy index (0 for PRO/NGL/NLL, 0..ML_MAT_COUNT-1 for ML). Non-ML nodes always get `iter_index=0`.

`body_label` is already on `GraphNode` (carries "PRO"/"ML-1"/"ML"/"NGL"/"NLL"). No new field needed for body annotation; `iter_index` disambiguates ML iter copies. `position` (SchedulePosition) stays as a per-node attribute for diagnostic consumers but is no longer the primary ordering basis in Phase 2.

### D — `byte_key_writers` position-type upgrade

Currently `byte_key_writers.setdefault(bk, []).append((node, node.position))` — stores `(GraphNode, SchedulePosition)` pairs.

C3c upgrades to `byte_key_writers.setdefault(bk, []).append((node, node.unrolled_position))` — stores `(GraphNode, int)` pairs where the `int` is `unrolled_position`.

This matches the C3b docstring comment: "C3c will populate with `(GraphNode, unrolled_position: int)` pairs." The `DataflowGraph.byte_key_writers` docstring already states this contract. `SchedulePosition` stays on the node itself for diagnostics; it is not the lookup basis for Phase 2 ordering.

### E — SCC-boundary clearing semantics

The current code detects body-boundary transitions by comparing `prev_body_label != node.body_label` and clearing SCC entries. In the unrolled walk, ML iter copies have the same `body_label = "ML"` — so no spurious SCC clearing occurs between ML_iter[0] and ML_iter[1]. The SCC clearing only fires when transitioning from ML to NGL (body_label changes from "ML" to "NGL"), and from NGL to NLL. This is correct: SCC is a single-bit hardware register not preserved across loop iterations, but the ML-to-ML iter transition within the unrolled model is an artificial expansion (not a real loop-iteration boundary in the SCC sense). The plan (§1 constraints) says "barriers reset SCC's `latest_writer` entry — preserved." The mechanism (compare `prev_body_label`) will continue to work correctly for ML-to-NGL and NGL-to-NLL transitions. For ML iter copies, since they share `body_label = "ML"`, the transition check sees same label and does NOT clear SCC. This is correct behavior: across ML iter copies the SCC state is not physically reset.

Worked example (PRO → ML_iter[0] → ML_iter[1] → NGL → NLL):
- PRO → ML_iter[0]: `prev_body_label = "PRO"`, new `body_label = "ML"` → SCC cleared ✓
- ML_iter[0] → ML_iter[1]: `prev_body_label = "ML"`, new `body_label = "ML"` → NO clear (correct: SCC not reset mid-loop)
- ML_iter[1] → NGL: `prev_body_label = "ML"`, new `body_label = "NGL"` → SCC cleared ✓
- NGL → NLL: `prev_body_label = "NGL"`, new `body_label = "NLL"` → SCC cleared ✓

### F — Latest_writer walk in detail

The single-pass unrolled-stream walk (no per-body resets except SCC at body-boundary transitions):

```
for each record in unrolled_capture.records:  # PRO, ML_iter[0], ML_iter[1], NGL, NLL
    for local_idx, tagged_inst in enumerate(record.instructions):
        unrolled_pos = record.unrolled_start + local_idx
        node = make_node(tagged_inst, record.body_label, record.iter_index, unrolled_pos, ...)
        if prev_body_label is not None and node.body_label != prev_body_label:
            # clear SCC keys from latest_writer
        prev_body_label = node.body_label

        # Phase 2a: reads first
        for each read_resource in node.reads:
            for producer, overlap, intra_offsets, src_slot in _resolve_producers(read_resource, node, latest_writer):
                emit DataflowEdge(producer, node, ...,
                    producer_iter_index=producer.iter_index,
                    consumer_iter_index=node.iter_index,
                    producer_body_label=producer.body_label,
                    consumer_body_label=node.body_label,
                    producer_unrolled_position=producer.unrolled_position,
                    consumer_unrolled_position=unrolled_pos)

        # Phase 2b: writes second
        for each write_resource in node.writes:
            for bk in _byte_keys_for_resource(write_resource, name_to_idx=n2i):
                latest_writer[bk] = (node, write_resource, w_slot)
                byte_key_writers.setdefault(bk, []).append((node, unrolled_pos))
```

The NLL cross-body scenario from BPG#11: after ML_iter[0] and ML_iter[1] both process their PackB CVT writes, `latest_writer[('v', 31)]` etc. hold the ML_iter[1] PackB3 writers. When NLL's first MFMA consumer resolves its read of `('v', 31..34)`, `latest_writer` returns the ML_iter[1] PackB3 writers — correct cross-iter edge formed. The 192 BPG#11 extras should resolve.

This matches §3.2 of the plan exactly.

### G — Identity iter-blindness preserved

Per C3a (`abgv`), `TaggedInstruction.identity_for(body_label)` returns `(canonical_render, source_module_id, emission_ordinal)`. The same `TaggedInstruction` object shared across ML iter copies returns the same identity regardless of `body_label` argument (since the `canonical_render`, `source_module_id`, and `emission_ordinal` are frozen on the object). So:
- ML_iter[0] node: same `identity` as ML_iter[1] node (same `TaggedInstruction` object)
- ML_iter[0] node: different `unrolled_position` from ML_iter[1] node
- ML_iter[0] node: `iter_index = 0`; ML_iter[1] node: `iter_index = 1`
- Both ML iter copies share the same `position: SchedulePosition` (since they reference the same `TaggedInstruction` and `_make_node` builds `position = make_position(body_label, stream_index)` from the same slot data). This means `position` is NOT a unique key for iter copies — `unrolled_position` IS.

**Important implication:** `_make_node` currently calls `make_position(body_label, stream_index)` where `stream_index` comes from `assign_stream_indices_for_body`. For ML iter copies the `tagged_inst` is shared — both copies produce the same `stream_index`. Both produce the same `SchedulePosition`. The `position` field on both iter-copy nodes is the same object value. This is fine: `position` is a diagnostic annotation on the node; `unrolled_position` is the canonical ordering key for the unrolled walk.

### H — The 11 break-list tests and expected failure delta

From `UNROLLED_VALIDATION_PLAN.md §6` and `6QIB_DESIGN.md §2.1`:

| Test | Classification | C3c behavior |
|---|---|---|
| `test_validate_pack_graph.py::test_pack_before_swap_orderinverted` | (a) | PASSES unchanged — pure within-body reorder; unrolled walk still detects it |
| `test_ValidateSCCoverlap.py::*` (5 tests) | (a) | PASS unchanged — SCC-clearing at body-boundary preserved |
| `test_validate_gr_not_too_early_graph.py::TestGRNotTooEarlyDtlPlusLdsBufGraph::test_negative_one_prev_iter_lr0_not_drained` | (a) | PASS unchanged |
| `test_cross_subiter_alu_carveout_real_kernel.py::test_real_kernel_validates_clean_with_carveout_engaged` | (b) | Re-fixture: now passes via unrolled walk resolving the cross-iter edge naturally; no exemption invoked |
| `test_cross_subiter_alu_carveout_real_kernel.py::test_real_kernel_neutralized_carveout_surfaces_768_pack3_mfma_failures` | (b) | Re-fixture or delete: the unrolled walk resolves the edges correctly, so "neutralizing" the carveout no longer surfaces failures |
| `test_cross_subiter_pack_artifact.py::test_carveout_suppresses_artifact_and_neutralization_surfaces_it` | (b) | Re-fixture |

Note: `test_cross_subiter_alu_carveout_real_kernel.py` does not have `xfail` markers per code inspection — these tests are currently FAILING (from C1's exemption deletion). C3c is expected to make the first one pass (clean validate from unrolled walk) and the second needs re-examination. The `test_cross_subiter_pack_artifact.py` companion tests (`test_artifact_present_in_default_graph`, `test_correct_edge_present_in_cms_graph`) are borderline (b)/(c) — re-evaluate against actual unrolled-walk output.

End-of-teym failure count: "22 FAILED + 2 ERROR" per the bead description. C3c should resolve the cross-iter pack-MFMA failures that the exemption deletion surfaced (the tests pinning the exemption's silencing behavior). The exact count resolved by C3c depends on which tests were failing because of the exemption deletion vs. structural model issues (C3d/C3e). The accept condition is: "the 11 break-list tests pass under the new walk."

Tests that will STAY RED after C3c: any test relying on `edge_keys()` byte-key comparison (C3d), `compare_graphs` symmetric direction (C3e), or `diagnose_missing_edge` Phase 0/1 migration (C3f).

### I — test_dataflow_graph_builder.py:519-523 re-fixture

Current test (`test_identity_stable_across_reorderings`):
```python
ids_a = [n.identity for n in g_a.nodes if n.category == "LRA0"]
ids_b = [n.identity for n in g_b.nodes if n.category == "LRA0"]
assert len(ids_a) == 1
assert len(ids_b) == 1
assert ids_a[0] == ids_b[0]
```

The test constructs a `FourPartCapture` via `_wrap(cap_a/cap_b)` where the ML body has one LRA0 instruction. After C3c, `build_dataflow_graph` walks the unrolled stream which includes ML_iter[0] AND ML_iter[1]. Both iter copies reference the same `TaggedInstruction`, producing two `GraphNode` objects with the same `identity`. So `ids_a` will have `ML_MAT_COUNT = 2` entries, all identical.

Re-fixture decision:
- `len(ids_a) == ML_MAT_COUNT` — correct count reflecting unrolled walk
- `ids_a == ids_b` — correct: all entries in both lists are identical (same identity, iter-blind)

Use `ids_a == ids_b` (list equality — both are `[ident, ident]`). This is stronger than `set(ids_a) == set(ids_b)` because it also verifies count. The test is pinning: "same identity on both iter copies, and both graphs produce the same count."

The re-fixture must also import `ML_MAT_COUNT` from `ScheduleCapture` in the test file.

### J — Edge annotation fields for diagnostics

The six fields are added to `DataflowEdge` as optional fields with sensible defaults:
- `producer_iter_index: int = 0` — set to `producer.iter_index` at edge formation
- `consumer_iter_index: int = 0` — set to `node.iter_index` at edge formation
- `producer_body_label: str = ""` — set to `producer.body_label` at edge formation
- `consumer_body_label: str = ""` — set to `node.body_label` at edge formation
- `producer_unrolled_position: int = -1` — set to `producer.unrolled_position` at edge formation
- `consumer_unrolled_position: int = -1` — set to `node.unrolled_position` (= current `unrolled_pos` in the walk) at edge formation

They are positional-default (`= value`) rather than required constructor args, for backwards compatibility with any test code that constructs `DataflowEdge` directly. They do NOT appear in `edge_keys()`. They are for human display in failure formatters and for Phase 1 order-check arithmetic (C3f will consume `unrolled_position` for cross-body ordering; the values are available now for forward compatibility).

### K — NLL structure preserved

Per `NLL_STRUCTURE_INVESTIGATION.md`: NLL is single-invocation. `FourPartCapture.n_ll = {0: body}`. The `UnrolledCapture.from_four_part_capture` already handles this correctly — it appends one `UnrolledIterRecord` for NLL with `iter_index=0`. The unrolled stream walks NLL's instructions once. No new "NLL-iter materialization" is needed or appropriate. NLL's internal subiter chain (`for uIdx in range(LoopIters)` inside `_noLoadLoopBodyDefault`) is already physicalized within the single captured body. Confirmed: NLL appears once in `UnrolledCapture.records`.

---

## §3 Design

### New `build_dataflow_graph` algorithm

**Entry point change:** After seeding `captures` (unchanged), call `UnrolledCapture.from_four_part_capture(four_part_capture)` to obtain the unrolled timeline. This replaces the `_BODY_BUILD_ORDER` iteration in Phase 1.

**Phase 1 — node construction over the unrolled stream:**

```python
unrolled = UnrolledCapture.from_four_part_capture(four_part_capture)
nodes_list = []
nodes_per_body = {label: [] for label in _BODY_BUILD_ORDER}

for record in unrolled.records:
    body = captures.get(record.body_label)
    if body is None:
        continue

    # Per-body stream_index assignment for SchedulePosition (unchanged).
    stream_idx_by_id = assign_stream_indices_for_body(body.instructions)

    # For ML iter copies > 0, reuse the same stream_idx_by_id from iter 0
    # (same TaggedInstruction objects). Only compute once per unique body.
    dataflow_ti_ids = {id(ti) for ti in data_flow_instructions(body)}

    for local_idx, tagged_inst in enumerate(record.instructions):
        unrolled_pos = record.unrolled_start + local_idx
        stream_index = stream_idx_by_id.get(id(tagged_inst), 0)

        node = _make_node(tagged_inst, record.body_label, stream_index, arch_profile)
        node.unrolled_position = unrolled_pos
        node.iter_index = record.iter_index

        # Sidecar: all nodes including sync ops.
        # For ML iter copies, only the first copy (iter_index=0) populates
        # nodes_per_body — the sidecar is body-keyed (not iter-keyed) and
        # the wait/barrier helpers use body._graph_nodes for per-body window
        # walking. Duplicate sidecar entries for ML iter>0 would corrupt
        # those helpers. Iter copies > 0 are in the unrolled stream for
        # dataflow purposes only.
        if record.iter_index == 0:
            nodes_per_body[record.body_label].append(node)

        if id(tagged_inst) in dataflow_ti_ids:
            nodes_list.append(node)

# Stash per-body sidecars (unchanged timing/wait helpers rely on this).
for label in _BODY_BUILD_ORDER:
    body_cap = captures.get(label)
    if body_cap is not None:
        body_cap._graph_nodes = nodes_per_body[label]
```

**Phase 2 — single `latest_writer` walk:**

The walk is now over `nodes_list` sorted by `unrolled_position` (not `SchedulePosition`):

```python
sorted_nodes = sorted(nodes_list, key=lambda n: n.unrolled_position)
```

The SCC-clearing logic compares `prev_body_label` against `node.body_label`. Body-boundary transitions (PRO→ML, ML→NGL, NGL→NLL) still clear SCC. ML iter copies don't trigger clearing (same `body_label = "ML"`). All other Phase 2 logic is identical — `latest_writer`, `_resolve_producers`, `byte_key_writers` — with two changes:
1. `byte_key_writers` stores `(node, node.unrolled_position)` (int) instead of `(node, node.position)` (SchedulePosition).
2. Each emitted `DataflowEdge` gets the six new annotation fields set from `producer.iter_index`, `producer.body_label`, `producer.unrolled_position`, and the equivalent consumer fields.

### New GraphNode fields

Add to `GraphNode` dataclass (with defaults so existing construction sites don't break):
```python
unrolled_position: int = -1    # set by build_dataflow_graph Phase 1
iter_index: int = 0            # set by build_dataflow_graph Phase 1; 0 for non-ML
```

### New DataflowEdge fields

Add to `DataflowEdge` dataclass (with defaults):
```python
producer_iter_index: int = 0
consumer_iter_index: int = 0
producer_body_label: str = ""
consumer_body_label: str = ""
producer_unrolled_position: int = -1
consumer_unrolled_position: int = -1
```

### byte_key_writers position-type upgrade

Change from `(GraphNode, SchedulePosition)` to `(GraphNode, int)`. The `int` is `node.unrolled_position`. The `DataflowGraph.byte_key_writers` docstring already anticipates this: "C3c will populate with `(GraphNode, unrolled_position: int)` pairs."

---

## §4 Phase-by-phase migration map

### Phase 1 today → C3c

| Today | C3c |
|---|---|
| Iterate `_BODY_BUILD_ORDER` | Iterate `unrolled_capture.records` |
| One `GraphNode` per ML instruction | One `GraphNode` per iter copy per ML instruction |
| `nodes_per_body[label].append(node)` always | `nodes_per_body[label].append(node)` only for `iter_index == 0` (sidecar correctness) |
| `body._graph_nodes = nodes_per_body[label]` per body loop | Same, but done after all records processed |
| No `unrolled_position`, no `iter_index` | Both set on every node |

### Phase 2 today → C3c

| Today | C3c |
|---|---|
| `sorted_nodes = sorted(nodes_list, key=lambda n: n.position)` | `sorted_nodes = sorted(nodes_list, key=lambda n: n.unrolled_position)` |
| Body-boundary SCC clearing via `prev_body_label` | Same mechanism; ML iter copies don't trigger (same `body_label`) |
| `byte_key_writers.append((node, node.position))` | `byte_key_writers.append((node, node.unrolled_position))` |
| `DataflowEdge(producer, node, ...)` — no annotation fields | `DataflowEdge(producer, node, ..., producer_iter_index=..., consumer_iter_index=..., ...)` |

The Phase 2 SCC-clearing logic is byte-identical to today except the sort key change. The `latest_writer` logic, `_resolve_producers` call, and `byte_key_writers` update are all preserved.

---

## §5 Call sites and integrations

### Import change in `CMSValidator.py`

Add `UnrolledCapture` to the import from `ScheduleCapture`:
```python
from Tensile.Components.ScheduleCapture import (
    ...
    UnrolledCapture,
    ...
)
```

### `_make_node` signature — no change needed

`_make_node(tagged_inst, body_label, stream_index, profile)` signature is unchanged. The new fields (`unrolled_position`, `iter_index`) are set on the returned node AFTER `_make_node` returns — they're instance attributes, not constructor args (because `_make_node` doesn't know `unrolled_position` at call time). Alternatively, add optional kwargs to `_make_node` — either approach is clean. Setting post-construction is simpler.

### `DataflowGraph.all_nodes_in_order` — no change

This property iterates `body._graph_nodes` via `_BODY_BUILD_ORDER`. Since C3c populates `nodes_per_body[label]` only for `iter_index == 0` (same as today), `all_nodes_in_order` continues to yield one copy of each node per body. The wait/barrier helpers that consume it are unaffected.

### `DataflowGraph.queue_depth_at` / `producer_queue_position` — no change

Both use `all_nodes_in_order` which is unchanged.

### `DataflowGraph.edge_keys()` — no change in C3c

Still uses `(e.producer.identity, e.consumer.identity, ...)`. C3d migrates this.

### `compare_graphs` / `diagnose_missing_edge` — no change in C3c

These consume `edge_keys()` and the `DataflowGraph.nodes` list. The `nodes` list now has more entries (ML_iter[0] + ML_iter[1] for each ML instruction) but the existing consumers iterate edges (not nodes directly) in the comparison path. C3f migrates the Phase 0/1 lookup logic.

### `CMSValidator.py` xj16 site — no change in C3c

The `build_dataflow_graph` call at line 4770-4771 is unchanged:
```python
ref_graph = build_dataflow_graph(context.default_capture)
subj_graph = build_dataflow_graph(context.cms_capture)
```

The function signature is unchanged; the internal rewrite is transparent.

### `KernelWriter.py` — no change

`build_dataflow_graph` is called from `CMSValidator.py` only via the inline xj16 site. `KernelWriter.py` only calls into the validator indirectly through `CMSValidator.validate_cms_schedule`. No changes needed in `KernelWriter.py` for C3c.

---

## §6 Re-fixture work

### Primary: `test_dataflow_graph_builder.py:519-523`

Test: `TestBasicDataflow::test_identity_stable_across_reorderings`

Before C3c:
```python
ids_a = [n.identity for n in g_a.nodes if n.category == "LRA0"]
ids_b = [n.identity for n in g_b.nodes if n.category == "LRA0"]
assert len(ids_a) == 1
assert len(ids_b) == 1
assert ids_a[0] == ids_b[0]
```

After C3c:
```python
from Tensile.Components.ScheduleCapture import ML_MAT_COUNT

ids_a = [n.identity for n in g_a.nodes if n.category == "LRA0"]
ids_b = [n.identity for n in g_b.nodes if n.category == "LRA0"]
assert len(ids_a) == ML_MAT_COUNT    # two iter copies of the ML LRA0
assert len(ids_b) == ML_MAT_COUNT
assert ids_a == ids_b                # all entries equal (iter-blind identity)
```

Justification: `ids_a == ids_b` (list equality) is stronger than `set(...)` and correctly pins that all `ML_MAT_COUNT` iter copies carry the same identity, and that count matches across captures.

### Secondary: break-list tests in C3c scope

At re-fixture time, classify each failure as (b) or (c):

**`test_cross_subiter_alu_carveout_real_kernel.py::test_real_kernel_validates_clean_with_carveout_engaged`**: After C3c, `compare_graphs` with the unrolled walk should find zero mismatches on BPG#11 (the cross-iter edges resolve to correct producers on both sides). Re-fixture to assert `failures == []` without the monkeypatch on `GraphNode.subiter` — the carveout is no longer needed because the unrolled walk resolves the dataflow correctly.

**`test_cross_subiter_alu_carveout_real_kernel.py::test_real_kernel_neutralized_carveout_surfaces_768_pack3_mfma_failures`**: This test asserts that neutralizing the carveout SURFACES 768 failures. After C3c, neutralizing the carveout (forcing all nodes to `subiter=0`) no longer surfaces failures from the cross-iter pipelining — the unrolled walk handles those. If other failures surface (real bugs), file as (c). If zero failures, delete the test (its premise is obsolete).

**`test_cross_subiter_pack_artifact.py`**: Re-evaluate each of the three methods against actual unrolled-walk output. The key question is whether `test_artifact_present_in_default_graph` and `test_correct_edge_present_in_cms_graph` are pinning correct cross-iter dataflow (which should now match between default and CMS under the unrolled walk) or genuine bugs.

### Sidecar/wait-coverage tests

If any test relies on `n.position` ordering in the `nodes` list (rather than `n.unrolled_position`), it may fail because the sort key changed. Scan `test_dataflow_graph_builder.py` for assertions on `g.nodes` ordering. The test at `test_identity_stable_across_reorderings` is the known one; others should be checked during implementation.

---

## §7 Step-by-step implementation order

1. **Add `unrolled_position` and `iter_index` fields to `GraphNode`** (default -1 and 0). Single-line dataclass additions.

2. **Add six annotation fields to `DataflowEdge`** (all with defaults). Single dataclass block.

3. **Add `UnrolledCapture` to the CMSValidator.py import** from `ScheduleCapture`.

4. **Rewrite `build_dataflow_graph` Phase 1**: replace the `for label in _BODY_BUILD_ORDER` loop with `for record in unrolled.records` iteration. Set `node.unrolled_position` and `node.iter_index` post-construction. Only populate `nodes_per_body[record.body_label]` for `record.iter_index == 0`.

5. **Rewrite `build_dataflow_graph` Phase 2**: change sort key from `n.position` to `n.unrolled_position`. Change `byte_key_writers.append((node, node.position))` to `byte_key_writers.append((node, node.unrolled_position))`. Add the six annotation fields to each `DataflowEdge(...)` construction call.

6. **Re-fixture `test_dataflow_graph_builder.py:519-523`**: change counts and comparison per §6.

7. **Run unit suite**; classify failures per (a)/(b)/(c). Re-fixture (b)-class tests. File new beads for any (c)-class real bugs.

8. **Write two new unit tests** (per §6.3 of the master plan, scoped to C3c):
   - Cross-iter live-in: ML iter 0 writes → ML iter 1 reads, edge formed with correct `producer_iter_index=0, consumer_iter_index=1`.
   - Cross-body live-in: PRO (or ML iter 1) writes → NGL/NLL reads, edge formed with correct body annotations.

---

## §8 Validation — expected failure delta

**End-of-teym baseline:** 22 FAILED + 2 ERROR.

**After C3c lands:**

Tests expected to resolve (go GREEN):
- `test_cross_subiter_alu_carveout_real_kernel.py::test_real_kernel_validates_clean_with_carveout_engaged` — unrolled walk resolves cross-iter edges correctly, clean validate.
- Any tests failing solely because the unrolled walk was missing (body-local walk produced wrong edge sets) that don't depend on `edge_keys()` byte-key comparison (C3d).
- `test_dataflow_graph_builder.py::test_identity_stable_across_reorderings` — re-fixtured.

Tests expected to stay RED:
- Tests depending on `edge_keys()` byte-key basis migration (C3d).
- Tests depending on `compare_graphs` symmetric direction (C3e).
- Tests depending on `diagnose_missing_edge` Phase 0/1 unrolled-position migration (C3f).
- The n7og xfail tests (`test_n7og_edge_keys_multifixture.py`) — these use `xfail strict=True` and will only flip to GREEN in C3h after all the pieces land.

C3c does NOT claim to make the validator fully GREEN. The bead acceptance is: "build_dataflow_graph walks one unrolled stream; no per-body dispatch in the walk." Validator state remains partly RED.

---

## §9 Risks / open questions

**Risk 1 — Sidecar correctness for ML iter copies:** `nodes_per_body[label]` must receive only `iter_index == 0` nodes; otherwise wait/barrier helpers that walk `body._graph_nodes` see duplicate entries, doubling wait-queue depth calculations. The implementation must guard on `record.iter_index == 0`.

**Risk 2 — `assign_stream_indices_for_body` called once vs. once per iter copy:** The function is keyed on `id(tagged_inst)`. For ML iter copies, all copies share the same `tagged_inst` objects, so the `stream_idx_by_id` dict computed for iter 0 is valid for all copies. Call it once per unique body and reuse across iter copies.

**Risk 3 — `data_flow_instructions` called once vs. once per iter copy:** Same as Risk 2. Compute `dataflow_ti_ids` once per unique body and reuse.

**Risk 4 — Break-list test (c)-class real bugs:** Some tests currently failing because of C1's exemption deletion may turn out to be real bugs the exemption was hiding. Per the standing rule, file each as a P0 bead. Do NOT re-fixture to make them pass.

**Risk 5 — `node.position` ordering vs. `unrolled_position` ordering for ML iter copies:** Both ML iter copies produce nodes with identical `SchedulePosition` (same `loop_index=1`, same `stream_index`). Sorting by `position` would be ambiguous (Python `sort` is stable but the relative order of iter copies is undefined). Sorting by `unrolled_position` is unambiguous — iter 0 always precedes iter 1. The sort-key change is load-bearing for correctness.

**Open question — `iter_delta_to` method on GraphNode:** `GraphNode.iter_delta_to(other)` computes `other.position.loop_index - self.position.loop_index`. For two ML iter copies (both `loop_index=1`), `iter_delta_to` returns 0. This is used by `Failure._iter_suffix` for diagnostic labels. Post-C3c, an `OrderInvertedFailure` involving an ML_iter[0] producer and ML_iter[1] consumer would show delta=0 in the label, which is technically wrong (it IS a cross-iter edge). C3f's Phase 1 migration should update `iter_delta_to` to use `unrolled_position` difference instead. File this as a note rather than a blocker for C3c — the diagnostic label is cosmetic.

**Risk 6 — ML-1 silently dropped (BLOCKER — rocm-libraries-tom4):** The unrolled stream does not include ML-1. The C3c Phase 1 replacement drops all ML-1 nodes, breaking the `TestCrossBodyQueueState` tests silently (the cross-body ML-1→ML edge disappears from the graph rather than raising). Must be resolved in §3 Design before implementation.

**Risk 7 — Breadth of re-fixture work underestimated (BLOCKER — rocm-libraries-2k8w):** Every fixture placing producer and consumer both in the ML body will produce ML_MAT_COUNT times the expected edges and nodes, breaking `len()==N` assertions throughout `test_dataflow_graph_builder.py`. §6 names only one test. The full scope must be enumerated before implementation.

---

## §10 New beads to file

Two blocking beads were filed during plan verification (2026-06-05):

- **rocm-libraries-tom4** (P0, bug): ML-1 body dropped from unrolled walk. Must resolve before implementation.
- **rocm-libraries-2k8w** (P0, bug): Re-fixture scope is larger than §6 states — all ML-body `len()==N` assertions in `test_dataflow_graph_builder.py` break under iter doubling. Full enumeration required before implementation.

Both are set as blockers for `rocm-libraries-1rsy`. The `iter_delta_to` diagnostic issue noted in §9 is cosmetic and belongs in C3f's scope (already tracked as `rocm-libraries-i190`). If (c)-class bugs surface during implementation, file new P0 beads per standing rules.

## §11 Verification Report (2026-06-05)

### Decision verdicts

| Decision | Verdict | Evidence |
|---|---|---|
| 1 — No per-body resets | CORRECT | `CMSValidator.py:2076` `latest_writer={}` once; line 2102 only clears SCC keys at body-boundary transitions. |
| 2 — Sidecar guard `iter_index==0` | CORRECT | `all_nodes_in_order` and queue-depth helpers walk `cap._graph_nodes` via `_BODY_BUILD_ORDER`; duplicates from iter>0 copies would double queue-depth counts. `iter_index==0` is the right gate. |
| 3 — Sort by `unrolled_position` | CORRECT | Two ML iter copies share identical `SchedulePosition` (same `loop_index=1`, same `stream_index`); sorting by `position` is ambiguous between them. `unrolled_position` is unambiguous and set before the sort. |
| 4 — `ids_a == ids_b` list equality | CORRECT | Both copies share the same `TaggedInstruction` object and therefore the same identity. List equality is stronger than set equality and correctly pins count. |
| 5 — Edge annotations post-construction | CORRECT | The six fields do NOT appear in `edge_keys()` (verified: `CMSValidator.py:1308-1311` shows only `producer.identity, consumer.identity, edge_kind, intra_operand_byte_offset, src_operand_slot, sink_operand_slot`). Post-construction setting is consistent with how `producer_write_byte_key` and `consumer_read_byte_key` are handled today (set in `edges.append(DataflowEdge(..., producer_write_byte_key=..., consumer_read_byte_key=...))` inline). The plan proposes setting the six new fields inline at the `DataflowEdge(...)` call — this is the correct idiomatic pattern, not truly "post-construction." |
| Edge_keys unchanged in C3c | CORRECT | `CMSValidator.py:1308-1311` confirmed; plan explicitly states it. |

### Surprising finding — DOWNGRADED TO RISK

The plan claims (§2A, §3, §F): "the 192 NLL extras fix because materializing ML twice puts iter[1]'s writers into latest_writer before NLL begins."

**This claim is unverifiable from the existing probe data and is likely wrong for the BPG#11 fixture.** Per `n7og_PROBE_REPORT.md`:

- The 192 extras are ALL in NLL body, not cross-body ML→NLL edges. They are NLL-internal producer-before-consumer vs consumer-before-producer ordering: SHADOW's NLL places CVT producers at stream_index 7-10 before the MFMA consumers at stream_index 14+; CMS's NLL places the same CVT producers at stream_index 84-88, after MFMA consumers at stream_index 0-11.
- Both CVT producers and MFMA consumers are WITHIN NLL in both graphs — no ML-body writer is involved in the 192 extra edges.
- Materializing ML twice creates cross-iter edges of the form `ML_iter[0].writer → ML_iter[1].reader`. That does not address intra-NLL ordering divergence where NLL's own CVT producers appear after NLL's MFMA consumers in the CMS ordering.
- The 192 extras are currently silenced by the cross-subiter ALU-producer exemption (N3 branch in `diagnose_missing_edge`), which fires because CMS labels the producers as `PackB3/PackA3` (subiter=3) while consumers are `MFMA` (subiter=0,1,2). Removing the exemption converts these to `OrderInvertedFailure` findings pointing at intra-NLL reordering. The unrolled walk does not change the intra-NLL stream order — it only adds ML_iter[1] nodes before NGL/NLL.

**The 192 NLL extras will NOT resolve from the C3c unrolled-walk change alone.** They require the full `OrderInvertedFailure` classification path (C3f Phase 1) to correctly characterize them as legitimate scheduler reorderings.

The plan's claim in §2A ("the 192 BPG#11 extras should resolve") must be struck. Replace with:

> RISK: The 192 BPG#11 NLL extras are intra-NLL stream-position divergences (CVT producers placed after MFMA consumers in CMS NLL ordering), not cross-ML-iter dataflow gaps. The unrolled walk adds ML iter copies for cross-iter pipelining correctness but does not change intra-NLL walk order. The 192 extras will remain divergences after C3c; they become correctly-classified `OrderInvertedFailure` instances only after C3f migrates Phase 1 to use `unrolled_position` for ordering. Add to §8 milestone: "produce graphs for BPG#11 fixture post-C3c and verify that the 192 extras are now routed to `OrderInvertedFailure` rather than silently absorbed by the exemption." This is a Phase 1 acceptance check for C3f, not C3c.

### ML-1 gap — BLOCKER confirmed

`_BODY_BUILD_ORDER = (BODY_LABEL_PROLOGUE, BODY_LABEL_ML_PREV, BODY_LABEL_ML, BODY_LABEL_NGL, BODY_LABEL_NLL)`. `UnrolledCapture.from_four_part_capture` covers only PRO, ML_iter[0..k], NGL, NLL. ML-1 is absent. The current `build_dataflow_graph` processes ML-1 as body #2 in `_BODY_BUILD_ORDER` and emits its nodes into `nodes_list`. The C3c unrolled-walk replacement would drop all ML-1 nodes silently. `rocm-libraries-tom4` blocks implementation.
