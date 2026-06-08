# 56e3 Option E Empirical Probe

## §0 The worry, step by step (with placeholders)

This section walks through the concrete scenario the probe was built to
falsify. Placeholders (`OP1`, `OP2`, `REG_A`, `REG_B`, generator
modules `MOD_X`, `MOD_Y`) replace assembly so the structure is easier
to follow.

### Setup — what `emission_ordinal` is, in one paragraph

Every captured instruction has three identity fields:
  - `canonical_render` — the rendered text of the instruction, including
    its register operands (e.g. "OP1 REG_A, REG_B"). Same operand
    spellings → same render. Different operand spellings → different
    render.
  - `source_module_id` — the generator module that emitted it
    (e.g. "MOD_X"). Two emissions from the same module share this; from
    different modules they differ.
  - `emission_ordinal` — a per-`(canonical_render, source_module_id)`
    counter assigned by `assign_emission_ordinals`. The counter walks
    instructions in `(slot.mfma_index, slot.sequence)` order — the
    SCHEDULE's slot lex order — and bumps once per render+source pair.
    First occurrence in that walk gets 0, second gets 1, and so on.

The crucial property: ordinal is assigned AFTER sorting by schedule
slots. The walk doesn't know which physical emission "originally" was
first; it just numbers them in the order the schedule lays them out.

### The scenario the worry imagines

Suppose generator module `MOD_X` emits two instructions with identical
render text — call them the "twins":

  twin_α : `OP1 REG_A, REG_B`  (from MOD_X)
  twin_β : `OP1 REG_A, REG_B`  (from MOD_X)

Both share `(canonical_render, source_module_id)` = `("OP1 REG_A, REG_B", "MOD_X")`.

SHADOW (default schedule) places them at slots:
  twin_α → (mfma_index=2, sequence=0)
  twin_β → (mfma_index=5, sequence=0)

`assign_emission_ordinals` walks in slot order, hits twin_α first, then
twin_β. So:
  twin_α.emission_ordinal = 0
  twin_β.emission_ordinal = 1

Now CMS schedules the SAME pair of physical instructions but pipelines
them, putting twin_β earlier and twin_α later:
  twin_α → (mfma_index=7, sequence=0)
  twin_β → (mfma_index=3, sequence=0)

CMS's `assign_emission_ordinals` walks in CMS slot order: twin_β
appears first (slot 3 < slot 7), then twin_α. So:
  twin_β.emission_ordinal = 0
  twin_α.emission_ordinal = 1

**Notice the flip.** On SHADOW the position-0 ordinal points to
twin_α; on CMS the position-0 ordinal points to twin_β. Same render,
same source module, swapped ordinal labels.

### Why the flip might break Option E

Option E proposes adding `(source_module_id, emission_ordinal)` to
`edge_keys()` so that two distinct producers writing the same byte
footprint don't collapse in set-diff.

Suppose twin_α writes byte-key BK1, and a consumer C reads BK1. SHADOW
has an edge:
  edge_SHADOW = (twin_α as producer, C as consumer, ... , producer_source="MOD_X", producer_ordinal=0)

CMS has the corresponding logical edge, but because the ordinals
flipped:
  edge_CMS = (twin_β as producer, C as consumer, ... , producer_source="MOD_X", producer_ordinal=0)

If twin_α and twin_β are PHYSICALLY DIFFERENT operations (different
reads or different writes despite identical render text — implausible
but worth checking), then edge_SHADOW with ordinal=0 means
"twin_α→C", edge_CMS with ordinal=0 means "twin_β→C", and these two
edges should NOT cancel in set-diff because they describe different
dataflows. Option E would falsely cancel them.

If twin_α and twin_β are SEMANTICALLY INDISTINGUISHABLE (same reads,
same writes — which they must be if their canonical_render is truly
identical AND their source_module_id matches), then the ordinal flip
swaps interchangeable operations. Edge_SHADOW and edge_CMS with
ordinal=0 describe equivalent dataflow either way. Option E is safe.

### What the probe checks

The probe enumerates every `(canonical_render, source_module_id)` group
with multiplicity > 1 in any body of any fixture, then asks: at every
position-i in the sorted SHADOW list and the sorted CMS list, are the
reads and writes of the two TaggedInstructions identical?

If yes for every group → flips are benign → Option E is safe.
If no for even one group → flips would corrupt edge_keys cancellation
→ Option E is unsafe and must be replaced (or hardened).

§3-§5 below report the actual measurements.

## §1 The question

Under CMS pipelining of TF32 + UsePLRPack schedules: are there cases
where the same `(canonical_render, source_module_id)` pair is emitted
multiple times within a single body, AND CMS schedules those emissions
in a different relative order than SHADOW — such that the same
`emission_ordinal` value at position-i in SHADOW points to a
semantically distinct rocisa instruction than the position-i instance
in CMS?

If yes → Option E (which would add `(source_module_id,
emission_ordinal)` into the `edge_keys` tuple to recover producer
discrimination) is unsafe: same (render, source_module_id,
emission_ordinal) on the two sides would refer to physically different
operations.

If no (i.e. position-i is always semantically interchangeable across
sides) → Option E is safe: ordinal flips, if they occur, swap
indistinguishable operations.

## §2 Methodology

`Tensile/Tests/unit/test_56e3_option_e_probe.py` runs `_build_shadow_cms_pair`
(same hook as `test_n7og_edge_keys_multifixture.py`) on two known
TF32+UsePLRPack fixtures that exercise the rotating T/X pack-buffer
registers. For each FourPartCapture body (PRO, ML-1, ML, NGL, NLL),
both SHADOW and CMS instruction lists are:

1. Sorted by `(slot.mfma_index, slot.sequence)` — the same sort
   `assign_emission_ordinals` uses.
2. Grouped by `(canonical_render, source_module_id)`.

For every group with multiplicity > 1 on either side, the probe
compares the position-i TaggedInstructions by reads/writes signature,
where the signature renders each `RegisterContainer`'s
`(regType, regIdx, regNum, regName.name, regName.offsets)` — content
fields, not Python identity.

Outcomes per group:

- `rw_match` — same multiplicity on both sides, and every position-i
  has identical reads/writes signature: Option E is safe for this
  group.
- `rw_differ` — same multiplicity, but at some position-i the reads or
  writes differ: Option E would index a semantically distinct
  operation at that ordinal on one side. Counter-example.
- `mult_differs` — SHADOW and CMS produce different numbers of
  emissions for the same key. Separately tracked.

## §3 Fixtures probed + raw counts

Fixtures: `bpg11-tf32-4x4-tn` (BPG#11) and `oplb-tf32-6x8-tn` (the OPLB
6x8 TN anchor). Both TF32 + UsePLRPack=True + UseMFMAF32XEmulation=True,
with rotating ValuA/B_T0/X0_I0 pack-buffer registers.

```
## Fixture: bpg11-tf32-4x4-tn
  body='ML'   cp=0: total_groups=184 mult>1=5 rw_match=2 rw_differ=0 mult_differs=3
  body='ML-1' cp=0: total_groups=184 mult>1=5 rw_match=2 rw_differ=0 mult_differs=3
  body='NGL'  cp=0: total_groups=168 mult>1=3 rw_match=1 rw_differ=0 mult_differs=2
  body='NLL'  cp=0: total_groups=138 mult>1=2 rw_match=1 rw_differ=0 mult_differs=1
  body='PRO' cp=None: total_groups=40  mult>1=0 rw_match=0 rw_differ=0 mult_differs=0

## Fixture: oplb-tf32-6x8-tn
  body='ML'   cp=0: total_groups=359 mult>1=5 rw_match=2 rw_differ=0 mult_differs=3
  body='ML-1' cp=0: total_groups=359 mult>1=5 rw_match=2 rw_differ=0 mult_differs=3
  body='NGL'  cp=0: total_groups=337 mult>1=4 rw_match=1 rw_differ=0 mult_differs=3
  body='NLL'  cp=0: total_groups=301 mult>1=3 rw_match=1 rw_differ=0 mult_differs=2
  body='PRO' cp=None: total_groups=70  mult>1=0 rw_match=0 rw_differ=0 mult_differs=0

## Overall counts
  total_groups=2140
  groups_with_multiplicity>1=32
  groups_rw_match (Option E safe)=12
  groups_rw_differ (Option E UNSAFE)=0
  groups_multiplicity_differs=20
  mult_differs_with_NON_NONE_source=0
```

## §4 Per-group analysis

### 4.1 Same-multiplicity groups (12/32)

Every single same-multiplicity, multi-emission group with a non-None
`source_module_id` is `rw_match`. Concretely these are:

- `s_add_u32 m0, m0, 4224` from `globalReadDoA_1` — multiplicity 3
  emissions per body, same physical (regType, regIdx, regNum)
  read/write tuples at every position-i on both sides.
- `s_add_u32 m0, m0, 4224` from `globalReadDoB_1` — same pattern.
- Other DTL/m0-update repeats from the same source modules.

CMS reorders these in `(mfma_index, sequence)` (e.g., SHADOW slots
`[(16,1), (19,1), (23,1)]` vs CMS `[(18,3), (20,3), (25,1)]`), so the
`assign_emission_ordinals` walk produces the SAME ordinal sequence
(0,1,2 on both sides) but pointing at the SAME content because each of
the three physical emissions has the same reads/writes shape — they are
indistinguishable except for which physical register holds the
m0-source operand, and that operand renders identically on both sides
once you go through `_render_register`'s content fields.

This is consistent with the `KernelWriter` source structure: the three
`s_add_u32 m0, m0, ...` lowerings in `globalReadDoA_1` each consume the
same per-iter symbolic operand binding, so all three physical
emissions resolve to identical RegisterContainer content. CMS shuffles
their wall-clock positions but cannot synthesize a fourth or substitute
a different operand.

### 4.2 Multiplicity-differs groups (20/32)

All 20 cases are `source_module_id=None`. The render strings are
exclusively raw scheduling primitives:

- `s_waitcnt lgkmcnt(0)` (SHADOW=2, CMS=4) — CMS injects more
  waitcnt fences for its pipelined LDS read/write rhythm.
- `s_nop 0` (SHADOW=4, CMS=0) — SHADOW pads MFMA spacing with
  explicit nops; CMS folds equivalent latency into mfma_count fields
  and elides the nop nodes.
- `s_barrier` (SHADOW=1, CMS=2) — CMS issues extra barriers to
  rendezvous its pipelined buffer rotation; SHADOW emits one.

`source_module_id=None` means these instructions were not threaded
through a named generator Module — they were emitted as bare
scheduling primitives by the bridge. Under Option E's keying basis
`(canonical_render, source_module_id, emission_ordinal)`, all
None-source emissions of the same render share a counter, so a
multiplicity divergence would in principle produce edge_key
disagreements.

However: SWaitCnt / SNop / SBarrier do not contribute reads/writes,
so they do not become endpoints of `raw_intrawave` /
`lds_raw_intrawave` edges (which is what `DataflowEdge` indexes). They
do appear in the LDS-reuse barrier-edge patterns
(`lr_to_gr_lds_reuse`, `gr_to_lr_lds_reuse` — `CMSValidator.py:1050`),
but those edges are keyed by the byte-keys of the LR / GR endpoints
(not the barrier), so multiplicity changes in the barrier nodes do not
collide on edge_keys either.

The probe found ZERO `rw_differ` and ZERO `mult_differs` with a
non-None `source_module_id`. Every cross-side multiplicity drift
sits in unsourced sync primitives that do not enter the edge-key
tuple as a producer or consumer.

## §5 Verdict

**Option E is SAFE.**

For the only TaggedInstructions that could form `producer` /
`consumer` endpoints in `DataflowGraph.edges` (those with a non-None
`source_module_id` and real reads/writes), CMS reordering within a
same-(canonical_render, source_module_id) group preserves position-i
semantic content across SHADOW and CMS. `emission_ordinal` values at
position-i index semantically interchangeable operations.

The user-hypothesized failure mode ("CMS pipelining swaps two
same-render+same-source emissions into different relative slots,
flipping their emission_ordinal values to point at different
operations") does not occur on the canonical TF32 + UsePLRPack
fixtures because the multi-emission same-source repeats are
themselves indistinguishable in content — they're the same lowering
template invoked once per per-iter binding, and CMS can only permute
their wall-clock positions, not substitute their operands.

## §6 Minimal counter-example

None. `groups_rw_differ` is 0 across both fixtures, all five body
labels, both CMS and SHADOW captures.

The `mult_differs` cases (`s_waitcnt lgkmcnt(0)`, `s_nop 0`,
`s_barrier`) are NOT counter-examples to Option E because:

1. They have `source_module_id=None`, so any cross-side ordinal
   drift among them is by-design under Option E's keying basis.
2. They do not appear as producer/consumer endpoints in the
   `(producer_write_byte_key, consumer_read_byte_key, ...)` 6-tuple
   that `edge_keys` builds — they carry no register reads/writes
   that resolve to byte-keys.

## §7 Recommendation

Stick with Option E. Add the `(source_module_id, emission_ordinal)`
projection to the edge_key tuple as planned for 56e3. The probe
provides empirical evidence on the two canonical TF32 + UsePLRPack
fixtures that this projection cannot introduce spurious mismatches
from CMS pipelining of multi-emission same-source-module instruction
groups.

If at some future point a CMS schedule is added that DOES
physically substitute operands within a multi-emission
same-(render, source_module_id) group — which would surface as a
non-zero `rw_differ` count in this probe — re-run
`test_56e3_option_e_probe.py` on the new fixture. The probe is
parameterized by `(fixture_id, kernel_config)` and adding a new
fixture is a one-line `_FIXTURES`-style edit.

Probe script: `Tensile/Tests/unit/test_56e3_option_e_probe.py`.
Run command: from `.tox/unit/lib/python3.11/site-packages`,
`pytest Tensile/Tests/unit/test_56e3_option_e_probe.py -v -s`.
