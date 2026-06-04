# Probe Plan — Disprove the SHADOW per-iter-duplication hypothesis

## Hypothesis under test

**H:** SHADOW's `_captureDefaultSchedule` walks per-iter source modules
(`PackCodeA[0]`, `PackCodeA[1]`, ...) and captures each iteration's pack
code separately, even when the same physical CVT instruction appears in
multiple iters. This produces multiple `GraphNode` Python objects for what
CMS treats as one physical instruction. The grouping in `_resolve_producers`
(`ScheduleCapture.py:1521`) — keyed by `(id(writer_node), id(write_res),
write_slot)` — then splits the consumer's read into N per-byte edges on
SHADOW while CMS sees 1 wide edge.

## Empirical anchor

Fixture: `_BPG_11_TF32_4X4_TN` (see
`Tensile/Tests/unit/test_n7og_edge_keys_multifixture.py`).
Reported divergence under 6-tuple byte-key matching:

- SHADOW: 647 edges; CMS: 471 edges
- Symmetric diff: 192 extra in SHADOW, 16 missing in SHADOW
- Of the 192 extras, 184 (96%) have numeric byte-keys; 8 (4%) have the
  `('v', -1)` sentinel
- 6-tuple residual mismatches: 30 (BPG#11), 90 (oplb-style)
- Pattern observed by the round-2 fix agent: SHADOW emits N per-byte edges
  with `intra=(0,)`, `(1,)`, `(2,)`, `(3,)`; CMS emits 1 wide edge with
  `intra=(0,1,2,3)`

## Falsification stance

The agent's job is to **try to disprove H**. Run every probe needed to
collect contradictory evidence. Only if every probe is consistent with H
and inconsistent with the alternatives can the hypothesis stand.

## Predictions

If **H is TRUE**, all of these should hold for the failing edges:

- **P1**: `len(shadow_graph.nodes) > len(cms_graph.nodes)` in the body
  where the failing edges live (likely `ML`)
- **P2**: For at least one failing edge's producer, multiple `GraphNode`
  objects in SHADOW share the same underlying `id(node.rocisa_inst)`
- **P3**: For the failing edge's consumer reading N bytes, SHADOW's
  `latest_writer[bk]` for the N bytes returns N distinct `(writer_node, write_res, write_slot)`
  triples; CMS returns one triple repeated N times
- **P4**: The TaggedInstruction stream captured for the SHADOW body
  contains the producer's `rocisa_inst` more than once; CMS contains it
  once

If **H is FALSE**, the divergence has a different mechanism. The
alternatives to specifically check:

- **A1**: Same GraphNode count, but `id(write_resource)` differs between
  captures (fresh RegisterContainer instances per `extract()` call)
- **A2**: Same GraphNode + same write_resource, but `write_slot` differs
  (operand rule yields different slot index on the two capture paths)
- **A3**: The producer is actually MULTIPLE distinct rocisa Instruction
  instances on BOTH sides (e.g. four `VCvtPkBF16F32` calls), and SHADOW's
  capture preserves them as separate while CMS coalesces them by emitting
  one composite instruction (would mean the divergence is at the
  emission-builder layer, not the capture-walk layer)

## Probes

### Probe 0 — Establish baseline

Build SHADOW + CMS captures for BPG#11. Verify they reproduce the
reported divergence:

```python
from Tensile.Tests.unit.test_n7og_edge_keys_multifixture import _BPG_11_TF32_4X4_TN
from Tensile.Tests.unit.cms_test_utils import _make_solution
# build & capture as the fixture does
# expected: SHADOW=647 edges, CMS=471 in body 'ML'
```

Confirm:
- Exact edge counts on each side
- Per-body breakdown
- Identity of body where the divergence lives (probably `ML`)

### Probe 1 — Node-count comparison (tests P1)

```python
for body in ['PRO', 'ML-1', 'ML', 'NGL', 'NLL']:
    sh_nodes = [n for n in shadow_graph.nodes.values() if n.body_label == body]
    cm_nodes = [n for n in cms_graph.nodes.values() if n.body_label == body]
    print(f"{body}: SHADOW={len(sh_nodes)}, CMS={len(cm_nodes)}, diff={len(sh_nodes)-len(cm_nodes)}")
```

**P1 holds** ⇔ SHADOW has strictly more nodes than CMS in body `ML`.

### Probe 2 — Same-rocisa-inst duplication (tests P2)

```python
from collections import Counter

for graph, label in [(shadow_graph, 'SHADOW'), (cms_graph, 'CMS')]:
    rocisa_ids = [id(n.rocisa_inst) for n in graph.nodes.values()]
    counts = Counter(rocisa_ids)
    dups = {k: v for k, v in counts.items() if v > 1}
    print(f"{label}: total nodes={len(rocisa_ids)}, unique rocisa_inst ids={len(set(rocisa_ids))}, dups={len(dups)}")
    if dups:
        # show first 5 with their categories + body_labels
        for rid, count in list(dups.items())[:5]:
            matching = [n for n in graph.nodes.values() if id(n.rocisa_inst) == rid]
            print(f"  rocisa_inst id={rid}: count={count}")
            for n in matching:
                print(f"    body={n.body_label}, cat={n.category}, slot={n.tagged_inst.slot}")
```

**P2 holds** ⇔ SHADOW has rocisa_ids with `count > 1`; ideally CMS has
none.

### Probe 3 — Latest_writer triples for one failing edge (tests P3)

Pick a specific failing edge from the 192 SHADOW extras. Print its
producer + consumer details. Then re-run `_resolve_producers`'s lookup
manually for the consumer's read bytes:

```python
# Find an extra-in-SHADOW edge whose category is pack→MFMA
shadow_edge_keys = shadow_graph.edge_keys()
cms_edge_keys = cms_graph.edge_keys()
extras = shadow_edge_keys - cms_edge_keys
# (pick one with prod=CVT, cons=MFMA, body=ML)

# For the chosen edge:
ed = ... # find the DataflowEdge object whose key matches
print("PRODUCER:")
print(f"  category={ed.producer.category}")
print(f"  body_label={ed.producer.body_label}")
print(f"  identity={ed.producer.identity}")
print(f"  rocisa class={type(ed.producer.rocisa_inst).__name__}")
print(f"  id(rocisa_inst)={id(ed.producer.rocisa_inst)}")
print(f"  id(node)={id(ed.producer)}")
print("CONSUMER:")
# ... same dump
print(f"intra_operand_byte_offset={ed.intra_operand_byte_offset}")
print(f"byte_keys producer side={...}, consumer side={...}")
```

Then re-run grouping for both sides:

```python
# For SHADOW: pull the latest_writer state at the consumer's body
# (re-run Phase 2 up to the consumer's position, then dump latest_writer
# for the consumer's read byte_keys)
# Same for CMS.

# Expected if P3: SHADOW yields N distinct (writer_node, write_res, write_slot)
# triples for the N bytes; CMS yields one triple repeated N times.
```

If reconstructing `latest_writer` mid-pipeline is impractical, the agent
can instrument `_resolve_producers` (temporarily, in a probe-only fork)
to record the writer_groups dict it builds for the specific consumer.

### Probe 4 — TaggedInstruction stream duplication (tests P4)

```python
for capture, label in [(shadow_capture, 'SHADOW'), (cms_capture, 'CMS')]:
    # For body 'ML':
    body_cap = capture.main_loop  # or whichever attribute
    rocisa_ids_in_stream = [id(ti.wrapped.rocisa_inst) for ti in body_cap.instructions]
    counts = Counter(rocisa_ids_in_stream)
    dups = {k: v for k, v in counts.items() if v > 1}
    print(f"{label} main_loop stream: total={len(rocisa_ids_in_stream)}, "
          f"unique={len(set(rocisa_ids_in_stream))}, dups={len(dups)}")
```

**P4 holds** ⇔ SHADOW's main_loop stream contains rocisa instances
with `count > 1` (the per-iter walking duplicated them).

### Probe 5 — Test A3 explicitly: are there N distinct CVT instances?

If P2 and P4 both come back negative — i.e. no duplicate rocisa_inst ids
— then H is disproven and A3 (or some other mechanism) is the cause.
Test A3:

```python
# Count how many distinct CVT rocisa instances exist in the body ML
# (on either side; should agree)
for graph, label in [(shadow_graph, 'SHADOW'), (cms_graph, 'CMS')]:
    cvts = [n for n in graph.nodes.values()
            if type(n.rocisa_inst).__name__.startswith("VCvt")
            and n.body_label == 'ML']
    print(f"{label}: ML body has {len(cvts)} VCvt nodes; "
          f"unique rocisa_inst ids={len(set(id(n.rocisa_inst) for n in cvts))}")
```

If both sides have the same number of distinct CVT instances but SHADOW
emits more edges, the divergence is somewhere else — likely in
`_OPERAND_RULES.extract` producing different shapes, or in the
`write_resource` Python identity.

### Probe 6 — A1 / A2 specifically: write_resource identity + write_slot

For the failing edge picked in Probe 3, find the producer GraphNode and
inspect its operand-rule output:

```python
from Tensile.Components.ScheduleCapture import _OPERAND_RULES

inst = ed.producer.rocisa_inst
for rule in _OPERAND_RULES:
    if rule.applies(inst, category=ed.producer.category):
        result = rule.extract(inst, category=ed.producer.category)
        print(f"rule={type(rule).__name__}, extract={result}")
        # If the result includes RegisterContainer objects, print their id
        # and check whether two calls return same or fresh instances
        result2 = rule.extract(inst, category=ed.producer.category)
        for r1, r2 in zip(result[2], result2[2]):  # writes
            print(f"  write resource id consistency: {id(r1) == id(r2)}")
        break
```

If `extract()` returns FRESH `RegisterContainer` instances each call,
that's a candidate mechanism for A1: SHADOW and CMS each call
`extract()` independently → different `id(write_resource)` for what
should be the "same" resource.

## Required outputs

The agent writes the probe report to:
`/home/alvasile/rocm-libraries/.worktrees/validator_long_term_plans/projects/hipblaslt/tensilelite/Tensile/Components/n7og_PROBE_REPORT.md`

Structure:

```markdown
# n7og Probe Report — Mechanism of SHADOW-vs-CMS edge granularity divergence

## Verdict
[H confirmed / H disproven / Inconclusive — needs further probe]

## Empirical baseline (Probe 0)
[Edge counts on both sides, per-body breakdown, name of failing body]

## P1 — Node-count comparison
**Expected if H true:** SHADOW nodes > CMS nodes in failing body
**Result:** [actual numbers]
**Verdict:** [P1 holds / P1 violated]

[same shape for P2, P3, P4, A1, A2, A3]

## Specific failing edge investigated (Probe 3)
[Producer + consumer dump with exact rocisa class, ids, categories,
body, byte_keys, intra]

[Latest_writer state for the consumer's read bytes on both sides —
the actual (writer_node, write_resource, write_slot) triples]

## Synthesis
[2-3 paragraphs: which prediction held, which violated, what mechanism
is actually responsible]

## Updates needed to 6QIB_DESIGN.md
[List the §0.3 / §2.1 claims that need correction based on findings]
```

## Constraints on the probe

- The probe code lives in scratch files (not committed). It can shadow-
  monkeypatch `_resolve_producers` if needed to log mid-pipeline state.
- Do not modify the validator's actual logic. Probes are read-only;
  monkeypatching is for diagnostic logging only.
- If a probe requires changing code to capture state, capture state via
  a sidecar log (e.g. write to `/tmp/n7og_probe.log`), don't bake it
  into the validator.
- Trust empirical evidence over the doc. If the doc claims "SHADOW
  emits 4 edges with `PackB0`" but the probe shows different categories
  or different counts, report what the probe actually shows.
- Capture EXACT trace values. Hex ids, full string identities, full
  byte_keys tuples. Don't summarize numerically — the goal is to make
  the mechanism visible.
