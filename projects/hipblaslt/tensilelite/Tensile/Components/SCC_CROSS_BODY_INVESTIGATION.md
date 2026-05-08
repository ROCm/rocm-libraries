# SCC cross-body edge investigation (rocm-libraries-so9m)

Branch: `users/alvasile/so9m-scc-investigation` off vlt @ `bb96c76a19`.

The bead asks whether `diagnose_missing_edge`'s manual suppression of
cross-body SCC edges is hiding a resolver bug, hiding a real defect, or
documenting a benign artifact. This memo follows the code and answers.

## 1. Where the suppression lives now

File / function / lines:
`projects/hipblaslt/tensilelite/Tensile/Components/CMSValidator.py`,
inside `diagnose_missing_edge` (defined at line 2456), the SCC branch is
at lines 2576–2606. The cross-body suppression itself is the two lines
2584–2585:

```python
ref_resource = ref_edge.resource
if getattr(ref_resource, "regType", None) == "scc":
    ...
    if p_node.body_label != c_node.body_label:
        return []
```

What it does: when `diagnose_missing_edge` is asked to classify a
reference edge whose resource is the SCC singleton, and the reference
edge's producer and consumer live in different `body_label`s
(ML-1/ML/NGL/NLL), the function returns the empty list — i.e. no
`Failure` is emitted, the missing edge is silently absorbed. Same-body
SCC misses fall through into the existing clobber-detection branch
(2586–2606) that searches `subj_graph` for an intervening SCC writer and
emits `OverriddenInputFailure`. The bead description cited a pre-br4
line near `:3517`; the current line is `:2584` after the post-br4 split
that moved `diagnose_missing_edge` from `ScheduleCapture.py` to
`CMSValidator.py`.

The pre-rewrite §7.9 of `VALIDATOR_DESIGN.md` (still present at lines
475–485 of the current doc) frames this suppression with: "SCC is
single-bit and not preserved across loop iterations by any compiler
convention, so cross-body SCC edges in the default graph are aliasing
artifacts of the per-byte resolver." The same framing is duplicated in
the inline comment at CMSValidator.py:2578–2583.

## 2. Per-byte resolver behavior on SCC

SCC enters the resolver via `_SCCRule` in
`Tensile/Components/ScheduleCapture.py:1428–1523`. `applies()`
(line 1461–1472) returns `True` for any rocisa instruction whose
`reads_scc` or `writes_scc` flag is set (the C++ flags landed by bead
`dzl`). `extract()` (line 1474–1523) returns the same
`(reg_reads, read_slots, reg_writes, write_slots)` 4-tuple shape as
every other rule, with the SCC singleton from
`rocisa.instruction.scc_resource()` appended to reads/writes per the
flags. The SCC singleton is built by the rocisa C++ side as a
`RegisterContainer` with `regType="scc"`, `regIdx=0`, `regNum=1`
(see `_NUMERIC_REG_FACTORIES` at ScheduleCapture.py:1695–1706 and
`scc_resource` referenced at line 1475).

The resolver's keying lives in `_byte_keys_for_resource`
(ScheduleCapture.py:1050–1100). For a numeric register
(`resource.regIdx >= 0`) the keys are
`tuple((rt, resource.regIdx + i) for i in range(count))`
(lines 1088–1089). For SCC this yields exactly one key: `(("scc", 0),)`
— a single byte slot, regType-discriminated so it never collides with
vgpr/sgpr.

`_resolve_producers` (ScheduleCapture.py:1103–1160) consumes those
byte-keys against the per-byte `latest_writer` dict and yields one
producer per `(writer_node, write_resource, write_slot)` group. There
is **no SCC special case anywhere in this path** — SCC is treated as a
1-byte numeric register sitting in its own regType namespace, and goes
through the exact same writer-grouping and `_intersection`-overlap
machinery as a single-byte vgpr would. The "SCC singleton" naming is
about hash/equality stability (so two `scc_resource()` calls collapse
to one byte_key), not about any behavioral specialization.

This is also exactly what the `_SCCRule` docstring promises
(ScheduleCapture.py:1451–1458): "the per-byte latest-writer resolver
(Phase 2 of build_dataflow_graph) then naturally emits SCC RAW edges
between producers and consumers, and an intervening SCC clobber becomes
the new latest writer that breaks the producer's edge to the later
consumer."

## 3. Cross-body trajectory

The cross-body question reduces to: does `latest_writer` carry SCC
writes from one body to the next?

The driving loop is in `build_dataflow_graph`,
`CMSValidator.py:1016–1096`. The relevant lines are:

```python
if nodes_by_identity:
    ...
    latest_writer = {}  # byte_key -> (writer_node, write_resource, write_slot)
    sorted_nodes = sorted(nodes_by_identity.values(), key=lambda n: n.position)

    for node in sorted_nodes:
        ...
        # Phase 2a — emit edges for reads against current latest_writer.
        for read_idx, read_resource in enumerate(wrapped.reads):
            ...
            for producer, overlap, intra_offsets, src_slot in _resolve_producers(
                read_resource, node, latest_writer, name_to_idx=n2i,
            ):
                ...
                edges.append(DataflowEdge(...))
        # Phase 2b — update latest_writer with this node's writes.
        for write_idx, write_resource in enumerate(wrapped.writes):
            ...
            for bk in _byte_keys_for_resource(write_resource, name_to_idx=n2i):
                latest_writer[bk] = (node, write_resource, w_slot)
```

Two things to notice:

1. `latest_writer = {}` is initialized **once**, at line 1039, before
   the per-node loop. There is no intervening
   `latest_writer.clear()` between bodies; grep over CMSValidator.py
   confirms it (the only references are at lines 1019, 1020, 1035,
   1039, 1069, 1083, 1096, 2503 — none clear it).
2. `sorted_nodes` is sorted by `node.position`, which is a
   `SchedulePosition(loop_index, vmfma_index, sub_index)` and
   `loop_index = BODY_LABEL_TO_LOOP_INDEX[body_label]` (see
   `make_position` at CMSValidator.py:418–426). The body order is
   ML-1 (0) → ML (1) → NGL (2) → NLL (3) (`_BODY_BUILD_ORDER` at line
   859). So `sorted_nodes` yields a single contiguous stream that
   crosses body boundaries silently, and `latest_writer` accumulates
   across every transition.

Combine that with §2's SCC byte_key `("scc", 0)`: if any node in body
ML writes SCC and any later node in body NGL reads SCC, the resolver
will (a) find the ML writer in `latest_writer`, (b) `_intersection` the
two SCC singletons (matches), (c) emit a `DataflowEdge` whose producer
sits in ML and whose consumer sits in NGL. That edge has
`resource.regType == "scc"` and crosses bodies. This is precisely the
edge that `diagnose_missing_edge`'s suppression at line 2584 is
designed to drop on the floor when it goes missing in the subject
graph.

So: cross-body SCC edges are emitted naturally and predictably by the
resolver, exactly as a side effect of its general per-byte mechanism
running over a one-byte resource that has no body scoping.

## 4. Verdict

**Explanation 1 is the correct cause.**

Deciding code:

- The resolver's keying (`_byte_keys_for_resource` at
  ScheduleCapture.py:1088–1089) treats SCC like any other 1-byte
  numeric register; the SCC byte_key is `("scc", 0)`, body-agnostic.
- The driver loop's `latest_writer = {}` initialization
  (CMSValidator.py:1039) sits OUTSIDE the body iteration; the dict is
  never cleared and the sort order at line 1040 walks all bodies as
  one stream.
- `_SCCRule.extract()` (ScheduleCapture.py:1515–1522) publishes SCC
  reads/writes through the same `(reads, read_slots, writes,
  write_slots)` channel as every other rule; nothing downstream
  recognizes SCC's "single-bit, not preserved" hardware semantic.

Hypothesis 2 (SCC really is preserved across iterations and the
suppression hides defects) does not match the code. There is no
compiler convention or kernel pattern that depends on SCC value
surviving across a loop iteration boundary — SCC is single-bit and any
intervening scalar ALU writes it. The cross-body edges the resolver
emits are not real dataflow; they are bookkeeping artifacts of the
per-byte mechanism.

Hypothesis 3 (some third explanation) is not needed. The mechanism is
straightforward: a generic byte-tracker that's unaware of the
single-bit-not-preserved semantic.

## 5. Recommendation

**Option A: fix the resolver, remove the downstream suppression.**

Two reasons to prefer A over B (document the suppression where it
sits):

1. The suppression in `diagnose_missing_edge` is in the wrong layer.
   By the time a missing edge reaches the diagnostician, the bogus
   cross-body SCC edge has already been built, hashed, threaded
   through `compare_graphs` keying, and counted in the
   reference-graph edge set. Every consumer of the dataflow graph
   (visualizations, edge-equality keys, future analyses) sees and has
   to deal with edges that don't exist. Suppressing only at the
   missing-edge classifier is an N-call-sites problem; the resolver
   layer is the single source.
2. The "single-bit, not preserved across loop iterations" semantic is
   a stable hardware fact. Encoding it once in `_byte_keys_for_resource`
   (or in the build_dataflow_graph driver) costs O(few lines) and
   eliminates a class of artifact, rather than a single instance of
   it. The cleanest spot is to clear SCC entries from `latest_writer`
   at body boundaries — same body still gets clobber detection
   (existing path at CMSValidator.py:2586–2606 stays intact); cross
   bodies stop emitting the edges in the first place.

A regression test would mirror the bead's hypothesis-3 fixture: a pair
of bodies where SCC is written in ML and read in NGL; assert that
`build_dataflow_graph` does not emit an SCC-resource edge between
them.

**Follow-up bead recommended.** The fix itself is out of scope for
this investigation bead. File a follow-up that:
- Adds an explicit body-boundary SCC reset (or equivalent) to the
  driver loop in `build_dataflow_graph`
  (`CMSValidator.py:1042`–`1096`).
- Removes the `diagnose_missing_edge` suppression at lines 2584–2585.
- Adds a regression test for cross-body SCC edge non-emission.
- Re-adds a §7.9 paragraph in `VALIDATOR_DESIGN.md` documenting the
  resolver's body-boundary policy for SCC.

Nothing in the investigation suggests urgent attention. Hypothesis 2
(SCC IS preserved and real defects are being hidden) was specifically
ruled out: the suppression is dropping artifacts the resolver itself
should never have emitted.
