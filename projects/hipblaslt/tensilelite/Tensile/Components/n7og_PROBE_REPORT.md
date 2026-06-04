# n7og Probe Report — Mechanism of SHADOW-vs-CMS edge granularity divergence

## Verdict

**H is DISPROVEN.**

H predicted that SHADOW's per-iter source-module walking duplicates GraphNodes for a single physical CVT instruction, and that `_resolve_producers` then groups by `id(writer_node)` so the consumer's wide read fans out into N narrow per-byte edges. Every prediction this hypothesis makes (P1, P2, P3, P4) is empirically falsified on the BPG#11 fixture:

- **P1 violated**: SHADOW and CMS have IDENTICAL node counts in every body. ML=20/20, NGL=28/28, NLL=136/136, total 184/184.
- **P2 violated**: ZERO duplicated `id(rocisa_inst)` values on either side. SHADOW and CMS both have 184 unique rocisa instances mapped 1:1 to 184 GraphNodes.
- **P4 violated**: Per-body `TaggedInstruction` streams have ZERO duplicate rocisa_inst ids on either side (SHADOW ML stream=192 unique/192, CMS ML stream=194 unique/194; SHADOW NLL=148/148, CMS NLL=146/146).
- **P3 vacuous**: There's no "N distinct (writer_node, write_res, write_slot) triples" inflation. For the failing MFMA consumer in SHADOW, the 4 byte_keys do resolve to 4 distinct writers — but those 4 writers are 4 GENUINELY DIFFERENT CVT instructions (each writes a different VGPR slot), not 4 GraphNodes wrapping the same rocisa_inst.

**The actual mechanism is stream-position-ordering divergence in the NLL body**, NOT capture duplication and NOT edge-emission granularity:

- The failing body is `NLL`, not `ML` as the plan anticipated (`ML` has 17 edges on both sides, identical; the entire 192-extra delta lives in NLL where SHADOW=552 / CMS=360 edges).
- In SHADOW NLL: the 4 PackB0 CVT producers run at stream_index 7,8,9,10 BEFORE the first MFMA consumer at stream_index 14, so `latest_writer[('v',31..34)]` is populated when the MFMA's read is resolved, and 4 narrow edges (one per byte) are emitted.
- In CMS NLL: the same physical CVT producers are tagged with category `PackB3` (mfma_index=33) and placed at stream_index 84,85,87,88 — AFTER the MFMA consumers at stream_index 0,3,7,11,44,47,53,56,... so `latest_writer[('v',31..34)]` is empty when the MFMA's reads are resolved, and ZERO edges are emitted into those consumers.

The "192 extra in SHADOW" edges are not granularity-narrowing artifacts. They are dataflow edges that exist in SHADOW because the default scheduler placed the producers before the consumers in NLL, and DO NOT exist in CMS because the CMS scheduler placed those same producers after their consumers in NLL stream order.

## Empirical baseline (Probe 0)

SHADOW: 647 edges (across 184 nodes)
CMS: 471 edges (across 184 nodes)
edge_keys diff: 192 extra in SHADOW, 16 missing in SHADOW (208 total mismatches — matches the figure in the test docstring exactly).

Per-body node counts (SHADOW / CMS):

| body | SH nodes | CM nodes | diff |
|------|---------:|---------:|-----:|
| PRO  | 0 | 0 | 0 |
| ML-1 | 0 | 0 | 0 |
| ML   | 20 | 20 | 0 |
| NGL  | 28 | 28 | 0 |
| NLL  | 136 | 136 | 0 |

Per-body edge counts (by producer.body_label):

| body | SH edges | CM edges |
|------|---------:|---------:|
| PRO  | 0 | 0 |
| ML-1 | 0 | 0 |
| ML   | 17 | 17 |
| NGL  | 78 | 94 |
| NLL  | 552 | 360 |

The failing body is **NLL** (the no-load-loop tail), not ML as the plan anticipated. ML is fully aligned at 17 edges on both sides.

Sample extra-in-SHADOW edge_keys (each pulled from the 192):

```
(('v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+14], v[vgprValuA_X0_I0+12], v[vgprValuA_X0_I0+13]', None, 0),
 ('v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[52:55]', None, 0),
 'raw_intrawave', (2,), 0, 1)

(('v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+3], v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+7]', None, 0),
 ('v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[12:15]', None, 0),
 'raw_intrawave', (3,), 0, 0)
```

Sample missing-in-SHADOW edge_keys (each pulled from the 16):

```
(('ds_read_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:192', None, 0),
 ('v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+8+4:vgprValuB_X0_I0+8+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[20:23]', None, 0),
 'raw_intrawave', (0, 1, 2, 3), 0, 0)
```

The missing-in-SHADOW edges are wide `(0,1,2,3)` LR→MFMA edges, all in NGL body — the granularity mirror of the SHADOW NLL extras.

## P1 — Node-count comparison

**Expected if H true:** SHADOW nodes > CMS nodes in the failing body.

**Result:** Both sides have IDENTICAL node counts in EVERY body (PRO=0, ML-1=0, ML=20, NGL=28, NLL=136). Total 184 each.

**Verdict:** P1 violated. SHADOW does not duplicate GraphNodes per-iter; the per-body node populations are byte-identical to CMS.

## P2 — Same-rocisa-inst duplication

**Expected if H true:** SHADOW has rocisa_ids with count > 1; CMS has none.

**Result:**

| capture | total nodes | unique rocisa_inst ids | dup_count |
|---------|------------:|-----------------------:|----------:|
| SHADOW  | 184 | 184 | 0 |
| CMS     | 184 | 184 | 0 |

**Verdict:** P2 violated. ZERO GraphNodes wrap the same `id(rocisa_inst)` on either side. The capture does NOT emit multiple GraphNodes per CVT instance.

## P3 — Latest_writer triples for one failing edge

**Expected if H true:** For the failing edge's consumer reading N bytes, SHADOW's `latest_writer[bk]` returns N distinct `(writer_node, write_res, write_slot)` triples; CMS returns one triple repeated N times.

**Chosen failing edge** (from 192 NLL extras, restricted to producer.category startswith "Pack" and consumer.category == "MFMA"):

```
PRODUCER:
  category=PackB0
  body_label=NLL
  identity=('v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+0], v[vgprValuB_T0_I0+0], v[vgprValuB_T0_I0+1]', None, 0)
  rocisa class=VCvtPkF32toBF16
  id(rocisa_inst)=0x7696f74c6590
  id(node)=0x7696f7442610
  slot=SlotKey(subiter=0, slot_kind='pre_loop', mfma_index=-1, sequence=7)

CONSUMER:
  category=MFMA
  body_label=NLL
  identity=('v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[0:3]', None, 0)
  rocisa class=MFMAInstruction
  id(rocisa_inst)=0x7696f74734ed0
  slot=SlotKey(subiter=0, slot_kind='mfma', mfma_index=0, sequence=0)

edge_kind=raw_intrawave, intra=(0,), src_slot=0, sink_slot=0, resource=v[vgprValuB_X0_I0+0]
```

**Instrumented `_resolve_producers` trace** (monkey-patched on `CMSValidator._resolve_producers` — the validator's local binding, not `ScheduleCapture._resolve_producers`, because CMSValidator does a `from ... import _resolve_producers`):

SHADOW side (called when processing this MFMA consumer):

```
read_resource = v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3]
byte_keys (count 4):
  ('v', 31) -> writer_node_id=0x7696f7442610, write_res_id=0x7696f74da370, write_res=v[vgprValuB_X0_I0+0], write_slot=0
      writer cat=PackB0, body=NLL, canon=v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+0], v[vgprValuB_T0_I0+0], rocisa_id=0x7696f74c6590
  ('v', 32) -> writer_node_id=0x7696f7442710, write_res_id=0x7696f74d9ed0, write_res=v[vgprValuB_X0_I0+1], write_slot=0
      writer cat=PackB0, body=NLL, canon=v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+1], v[vgprValuB_T0_I0+2], rocisa_id=0x7696f74c6730
  ('v', 33) -> writer_node_id=0x7696f7442890, write_res_id=0x7696f74d9850, write_res=v[vgprValuB_X0_I0+2], write_slot=0
      writer cat=PackB0, body=NLL, canon=v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+2], v[vgprValuB_X0_I0+4], rocisa_id=0x7696f74c68d0
  ('v', 34) -> writer_node_id=0x7696f7442b10, write_res_id=0x7696f74da0b0, write_res=v[vgprValuB_X0_I0+3], write_slot=0
      writer cat=PackB0, body=NLL, canon=v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+3], v[vgprValuB_X0_I0+6], rocisa_id=0x7696f74c6a70
yielded 4 groups (one per byte, each a distinct writer + distinct rocisa instance):
  -> writer_id=0x7696f7442610, offsets=(0,), write_slot=0, writer_cat=PackB0  (rocisa_id=...c6590)
  -> writer_id=0x7696f7442710, offsets=(1,), write_slot=0, writer_cat=PackB0  (rocisa_id=...c6730)
  -> writer_id=0x7696f7442890, offsets=(2,), write_slot=0, writer_cat=PackB0  (rocisa_id=...c68d0)
  -> writer_id=0x7696f7442b10, offsets=(3,), write_slot=0, writer_cat=PackB0  (rocisa_id=...c6a70)
```

CMS side (called when processing the SAME identity MFMA consumer):

```
read_resource = v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3]
byte_keys (count 4):
  ('v', 31) -> None
  ('v', 32) -> None
  ('v', 33) -> None
  ('v', 34) -> None
yielded 0 groups
```

**Critical finding from this trace:** SHADOW's 4 writers are NOT four copies of the same instruction. Each `('v', 31..34)` writer is a SEPARATE `VCvtPkF32toBF16` instance with a SEPARATE rocisa_id (...c6590, ...c6730, ...c68d0, ...c6a70) writing to a DIFFERENT physical VGPR (`v[vgprValuB_X0_I0+0]`, `+1`, `+2`, `+3`). The consumer reads a 4-byte vgpr range `v[vgprValuB_X0_I0+0:+3]` and gets one byte from each of four distinct CVT instructions — `_resolve_producers`'s groupby key `(id(writer_node), id(write_res), write_slot)` correctly fans this out into 4 narrow edges because that is what the physical dataflow is: 4 narrow writes, 1 wide read.

**P3 is vacuously falsified.** There is no "same writer with N triples" inflation here. The 4 narrow edges are accurate per-byte ledger entries; CMS does not emit "1 wide edge" alternative — it emits ZERO edges (the writers don't exist in `latest_writer` at the consumer's moment).

## P4 — TaggedInstruction stream duplication

**Expected if H true:** SHADOW's main_loop stream contains rocisa instances with count > 1 (the per-iter walking duplicated them).

**Result:**

| capture | body | total | unique | dups |
|---------|------|------:|-------:|-----:|
| SHADOW  | ML   | 192 | 192 | 0 |
| CMS     | ML   | 194 | 194 | 0 |
| SHADOW  | NLL  | 148 | 148 | 0 |
| CMS     | NLL  | 146 | 146 | 0 |

**Verdict:** P4 violated. ZERO TaggedInstructions wrap the same rocisa_inst on either side, in either body. Per-iter source-module walking does NOT happen in this fixture, or if it does it is correctly de-duplicated before reaching `LoopBodyCapture.instructions`.

(Sidebar: SHADOW vs CMS stream sizes differ by 2 in both ML and NLL but the resulting node populations agree at 20 and 136 — the 2-entry deltas are non-dataflow stream entries excluded by the `data_flow_instructions` filter.)

## A1 — write_resource identity differs between captures

The instrumented trace shows that SHADOW's 4 entries in `latest_writer` have 4 distinct `write_res_id` values (`...da370`, `...d9ed0`, `...d9850`, `...da0b0`) — but these correspond to 4 distinct physical write resources (`v[vgprValuB_X0_I0+0]`, `+1`, `+2`, `+3`), not fresh duplicates of the same resource. CMS does not have ANY entries to compare against. A1 (fresh `RegisterContainer` instances per `extract()`) does not appear to be the mechanism; this is moot once the latest_writer is empty on the CMS side.

## A2 — write_slot differs between captures

Same as A1: CMS's latest_writer is empty for these byte_keys at the consumer's moment, so there's no slot-mismatch to investigate. SHADOW's 4 writers all use write_slot=0, which is the correct positional slot for the dst of `VCvtPkF32toBF16`. A2 falsified for this fixture.

## A3 — N distinct CVT instances exist on both sides

**Result (NLL body):**

| capture | NLL nodes | VCvt nodes | unique CVT rocisa_ids |
|---------|----------:|-----------:|----------------------:|
| SHADOW  | 136 | 64 | 64 |
| CMS     | 136 | 64 | 64 |

By class: SHADOW NLL = `{VCvtPkF32toBF16: 64, MFMAInstruction: 64, DSLoadB128: 8}`; CMS NLL is identical.

40 of the 64 CVT rocisa_ids are SHARED between SHADOW NLL and CMS NLL (same Python object identity — `id(rocisa_inst)`). 24 CVT rocisa_ids are unique to each side — these are the rocisa instances that ended up in a DIFFERENT body on the other side (the cross-body migration pattern that body-blind identity is supposed to absorb).

For one shared rocisa instance (`rocisa_id=0x7696f74c6c10`, `v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+7], v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+7]`):

| capture | category | slot_kind | mfma_index | sequence | identity (canonical_render, source_module_id, emission_ordinal) |
|---------|----------|-----------|-----------:|---------:|---|
| SHADOW  | PackB0   | mfma      | 0          | 6        | (`v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+7], v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+7]`, None, 0) |
| CMS     | PackB3   | mfma      | 33         | 1        | (`v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+7], v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+7]`, None, 0) |

**Identities match** (the 3-tuple is byte-equal). Category and slot DIFFER. This confirms that the SAME physical instruction is captured with different category/slot bookkeeping on the two paths — which is OK for `edge_keys()` (category and slot are not in the key) but matters for stream-position assignment (which is built from `SlotKey.mfma_index`/`sequence`).

**A3 conclusion:** Both sides have the same number of distinct CVT instances. The divergence is NOT in the capture's instance count; it is in the order those instances are placed in the stream and the category/slot labels they get assigned. The N-distinct-vs-1-composite story in the design doc does not apply to this fixture.

## Stream-position trace — the actual mechanism

Walking `graph.nodes.values()` in `SchedulePosition`-sorted order, all writes and reads of byte_keys `('v', 31)`-`('v', 34)`:

**SHADOW** (4 writes, 36 reads from those keys across all consumers):

```
Writes to ('v', 31)-('v', 34) in SHADOW:
  pos=(loop_index=3, stream_index= 7), body=NLL, cat=PackB0, bk=('v', 31)  v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+0]
  pos=(loop_index=3, stream_index= 8), body=NLL, cat=PackB0, bk=('v', 32)  v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+1]
  pos=(loop_index=3, stream_index= 9), body=NLL, cat=PackB0, bk=('v', 33)  v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+2]
  pos=(loop_index=3, stream_index=10), body=NLL, cat=PackB0, bk=('v', 34)  v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+3]

First MFMA read of those keys in SHADOW:
  pos=(loop_index=3, stream_index=14), body=NLL, cat=MFMA, bks={31,32,33,34}  v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0:+3], v[vgprValuA_X0_I0+0:+3]
```

Order: writes at 7,8,9,10; first read at 14. `latest_writer` is populated → 4 narrow edges emitted.

**CMS** (4 writes, 36 reads from those keys across all consumers):

```
Writes to ('v', 31)-('v', 34) in CMS:
  pos=(loop_index=3, stream_index=84), body=NLL, cat=PackB3, bk=('v', 31)  v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+0]
  pos=(loop_index=3, stream_index=85), body=NLL, cat=PackB3, bk=('v', 32)  v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+1]
  pos=(loop_index=3, stream_index=87), body=NLL, cat=PackB3, bk=('v', 33)  v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+2]
  pos=(loop_index=3, stream_index=88), body=NLL, cat=PackB3, bk=('v', 34)  v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+3]

First MFMA read of those keys in CMS:
  pos=(loop_index=3, stream_index= 0), body=NLL, cat=MFMA, bks={31,32,33,34}  v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0:+3], v[vgprValuA_X0_I0+0:+3]
  pos=(loop_index=3, stream_index= 3), body=NLL, cat=MFMA, ...
  pos=(loop_index=3, stream_index= 7), body=NLL, cat=MFMA, ...
  pos=(loop_index=3, stream_index=11), body=NLL, cat=MFMA, ...
  ...
```

Order: first MFMA read at stream_index=0; writes don't happen until stream_index=84,85,87,88. By the time the writes ARE processed, the MFMA consumers have already been processed — `latest_writer` is empty for those byte_keys when the reads are resolved, so ZERO edges are emitted into those MFMA consumers.

The 4 PackB3 writers DO eventually populate latest_writer (at stream_index 84-88) and DO produce edges into LATER PackB3→PackB3 self-consumers (24 edges) — those are the CMS-only entries that show up in the per-category breakdown.

## Per-body category breakdown (where the 192/16 deltas come from)

NLL body, edges aggregated by `(producer.category, consumer.category, edge_kind)`, only differing rows shown:

| (p_cat, c_cat, kind) | SH | CM | diff |
|---|---:|---:|---:|
| ('PackA1', 'MFMA', 'raw_intrawave') | 96 | 0 | +96 |
| ('PackB1', 'MFMA', 'raw_intrawave') | 96 | 0 | +96 |
| ('PackA3', 'PackA3', 'raw_intrawave') | 0 | 24 | -24 |
| ('PackB3', 'PackB3', 'raw_intrawave') | 0 | 24 | -24 |
| ('PackA1', 'PackA1', 'raw_intrawave') | 24 | 0 | +24 |
| ('PackB1', 'PackB1', 'raw_intrawave') | 24 | 0 | +24 |
| ('LRB0', 'PackB1', 'raw_intrawave') | 20 | 0 | +20 |
| ('LRB0', 'PackB0', 'raw_intrawave') | 0 | 20 | -20 |
| ('LRA0', 'PackA0', 'raw_intrawave') | 0 | 20 | -20 |
| ('LRA0', 'PackA1', 'raw_intrawave') | 20 | 0 | +20 |

Net for NLL: SH=552 / CM=360, matching subtotal = 272. The 192 extras break down exactly as `PackA0→MFMA: 96` + `PackB0→MFMA: 96`. The 16 missing-in-SHADOW come from NGL (LR3→MFMA: 16 missing).

NGL body, same shape:

| (p_cat, c_cat, kind) | SH | CM | diff |
|---|---:|---:|---:|
| ('LRA3', 'PackA0', 'raw_intrawave') | 20 | 0 | +20 |
| ('LRB3', 'PackB0', 'raw_intrawave') | 20 | 0 | +20 |
| ('LRB3', 'PackB3', 'raw_intrawave') | 0 | 20 | -20 |
| ('LRA3', 'PackA3', 'raw_intrawave') | 0 | 20 | -20 |
| ('LRA3', 'MFMA', 'raw_intrawave') | 0 | 8 | -8 |
| ('LRB3', 'MFMA', 'raw_intrawave') | 0 | 8 | -8 |

The category-pair diffs are not symmetric in a "narrow-vs-wide" sense; they are SHADOW and CMS routing reads to producers of DIFFERENT category labels, AND CMS reaching producers that SHADOW does not (LR3→MFMA in NGL has 8 CMS-only edges per operand because CMS places the LR3 producers in a position visible to the MFMA consumer; SHADOW does not).

## Sidebar finding — name_to_idx confirms it is NOT the udqg sentinel pattern

Both SHADOW and CMS NLL `name_to_idx` have the same bindings for the Valu* names:

| name | SHADOW NLL | CMS NLL |
|------|-----------:|--------:|
| ValuA_X0_I0_BASE | -1 | -1 |
| ValuB_X0_I0_BASE | 31 | 31 |
| ValuA_X0_I0 | -1 | -1 |
| ValuA_T0_I0 | 76 | 76 |
| ValuB_X0_I0 | 31 | 31 |
| ValuB_T0_I0 | 92 | 92 |

`ValuA_X0_I0 -> -1` is a separate bug (resolves to `('v', -1)` sentinel for any A-side read), present on BOTH sides and therefore CANNOT be the explanation for the SHADOW-vs-CMS divergence — both sides see the same sentinel. The udqg bead text in the test file (`name_to_idx is MISSING bindings for the rotating ValuA/B_T0_I0 / ValuA/B_X0_I0 pack-buffer registers`) is at least partially incorrect for THIS fixture as observed in THIS worktree at this branch tip — bindings are present, just resolved to a sentinel base for the A side. The 4% sentinel-edge fraction noted in the `n7og_CORRECTNESS_REPORT.md` figure quoted in the plan is consistent with sentinel being a minor contributor and the dominant 96% (184 of 192 extras) being the stream-ordering mechanism documented above.

## Synthesis

The probe falsifies the SHADOW per-iter-duplication hypothesis on every prediction it makes. There is no GraphNode duplication, no TaggedInstruction stream duplication, no "same rocisa_inst with N triples" inflation. SHADOW and CMS each capture exactly 184 GraphNodes wrapping 184 unique rocisa instances, and the per-body counts agree byte-for-byte. The `_resolve_producers` grouping mechanism is not the source of the 192 extras.

The actual mechanism is at the SchedulePosition (stream-index) layer. The CMS scheduler reorders the NLL body so that 64 MFMA consumers appear in the stream BEFORE the 64 CVT producers that write the registers those MFMAs read. The validator's `build_dataflow_graph` processes nodes in stream-sorted order and keeps a `latest_writer` dict that only records writes *seen so far* in the walk. When a consumer is reached, `_resolve_producers` looks up `latest_writer[bk]` for each byte the consumer reads — and on the CMS side those entries are empty because the producers haven't been processed yet. So no edges are emitted into the CMS NLL MFMAs from the CMS NLL CVTs at all. SHADOW emits 192 such edges because its default-scheduler ordering puts the CVTs first.

The 16 missing-in-SHADOW edges in NGL are the same mechanism mirrored: there CMS happens to place LRA3/LRB3 producers BEFORE the LRA3→MFMA / LRB3→MFMA consumers, so CMS emits those edges and SHADOW does not. The overall picture is "two different schedulers placed the same instructions into different positions in their respective NLL/NGL streams; the dataflow graph's stream-position-driven `latest_writer` walk produces different edge sets as a result." This is a SCHEDULER REORDERING DIVERGENCE measured by `compare_graphs`, not a CAPTURE-LAYER granularity divergence.

Implication for the design doc: §0.3 / §2.1's "SHADOW emits 4 per-byte edges; CMS emits 1 wide edge" framing is not what the data shows on this fixture. The data shows "SHADOW emits 4 per-byte edges; CMS emits 0 edges." The 30 / 90 6-tuple residuals listed in the design doc are consistent with stream-position-driven edge presence/absence, not with `_resolve_producers` granularity selection. Approach (b) "capture-layer alignment" as described (coalescing per-byte writes into wide-edge writes) would not eliminate the residuals — there are no "wide edges" on the CMS side to align with; there are no edges at all.

## Updates needed to 6QIB_DESIGN.md

The empirical mechanism observed on the BPG#11 fixture contradicts the specific narrowness-vs-wide-edge framing in the design doc. The following claims need correction, listed with file:line and proposed correction:

1. **`6QIB_DESIGN.md:49`** — Claim: *"Why SHADOW and CMS might differ on this [per-byte vs wide edge granularity]: subtle differences in how the producer's resource is described at capture time. If the producer is two different instructions on one side and a single composite instruction on the other (e.g., two `VPackB32` vs one `VPackB128`), grouping differs → edge count differs."*
   - **Correction:** This is not what happens in the BPG#11 fixture. The producers ARE the same physical rocisa instances on both sides (40 of 64 CVT instances in NLL share `id(rocisa_inst)` directly across SHADOW and CMS); on the side where the edge count is lower, the producer-side `latest_writer` entries are absent at consumer-resolution time because the CMS scheduler placed the consumer earlier in the stream than the producer. The SHADOW-side 4 narrow edges arise because the four `VCvtPkF32toBF16` producers each write a distinct physical VGPR (`v[vgprValuB_X0_I0+0]`, `+1`, `+2`, `+3`), and the MFMA's wide read of `v[vgprValuB_X0_I0+0:+3]` legitimately receives one byte from each of four different writers. The narrow-edge representation is correct; there is no wide-edge alternative to coalesce to.

2. **`6QIB_DESIGN.md:131`** — Claim: *"30 mismatches remain. These come from a per-byte-vs-wide-edge asymmetry between SHADOW and CMS (§0.3)."*
   - **Correction:** The 30 (and 192 under the 6-tuple identity basis) residuals on BPG#11 come from a SHADOW-vs-CMS stream-position ordering divergence in the NLL body, where the CMS scheduler places MFMA consumers before their CVT producers in NLL stream order, causing `latest_writer`-driven edge formation to emit zero edges into those CMS consumers while SHADOW emits 192 (the producer-first ordering populates `latest_writer` correctly). The asymmetry is presence-vs-absence, not narrow-vs-wide.

3. **`6QIB_DESIGN.md:139-140`** — Worked example claim: SHADOW emits 4 per-byte edges with category `PackB0`, CMS emits 1 wide edge with category `PackB3`.
   - **Correction:** SHADOW emits the 4 per-byte edges as described. CMS does NOT emit a wide edge in their place — CMS emits ZERO edges for the same consumer's read, because the four producer writes are absent from `latest_writer` at the moment the consumer is processed (consumer stream_index=0, producer stream_indices=84,85,87,88 in NLL). The "1 wide edge" with `intra_offset=(0,1,2,3)` on the CMS side does not exist for this consumer; the CMS NLL stream has 0 edges into the matching MFMA consumer.

4. **`6QIB_DESIGN.md:142`** — Claim: *"Under set-diff with the 6-tuple key, all 4 of SHADOW's per-byte tuples differ from CMS's 1 wide tuple (the `intra_offset` field differs). That's 4-of-30 mismatches from this one edge alone."*
   - **Correction:** Under set-diff, all 4 of SHADOW's per-byte tuples differ from CMS's ZERO-tuple set for that consumer (the CMS set has nothing for those byte-keys). Each of the SHADOW 4 contributes 1 extra-in-SHADOW mismatch; CMS contributes 0 missing-in-SHADOW because no CMS edge exists with the same producer-identity / consumer-identity. The arithmetic still produces 4 extras from this one consumer, but the reason is "CMS produces no edge to compare against," not "CMS produces a different-shaped edge."

5. **`6QIB_DESIGN.md:144`** — Claim: *"This is a SHADOW-vs-CMS edge-granularity divergence at the capture layer — a different problem from the register-renaming false positives the byte-key swap was meant to fix. No edge-key tuple shape solves it; the fix has to align how the two sides emit edges (or how they coalesce per-byte writes into per-resource writes) at the `_resolve_producers` step."*
   - **Correction:** The divergence is at the SCHEDULER ORDERING + STREAM-POSITION layer, not at the per-byte vs per-resource coalescing layer. `_resolve_producers` is doing the correct thing in both captures: it emits one edge per `(writer_node, write_resource, write_slot)` group of bytes resolved from `latest_writer`. The difference is purely in which `latest_writer` entries exist at the moment the consumer is resolved. Aligning `_resolve_producers` granularity does not fix this; that would require either (i) extending `build_dataflow_graph` to do a two-pass walk where all writers are seen before any consumer is resolved (would lose order-detection), or (ii) genuinely detecting the order-inversion as a real scheduler defect (which is what `OrderInvertedFailure` is for, and the CMS reordering observed here may in fact be a legitimate scheduler defect masquerading as a granularity issue).

6. **`6QIB_DESIGN.md:198`** — Claim: *"The 30+90 6-tuple structural mismatches from §2.1 are NOT addressed [by Approach (a)] — they're SHADOW-vs-CMS edge-granularity divergences, not key-tuple issues. Compound key won't fix them."*
   - **Correction:** The 30+90 residuals are stream-position ordering divergences, not granularity divergences. Approach (a) (compound key with `ordinal_class`) likely WOULD shift their classification — if `ordinal_class` distinguishes "producer-before-consumer" from "consumer-before-producer", then the SHADOW 192 extras would still be extras (producer-before-consumer = present on SHADOW only because of the missing CMS edge), but `diagnose_missing_edge` would correctly route them as `OrderInvertedFailure` rather than absorbing them silently. The fix shape is the same as the design doc proposes for Approach (a); the docstring rationale for "why they remain" needs updating.

7. **`6QIB_DESIGN.md:207`** — Claim: *"The 6-tuple residual SHADOW emits 4 per-byte edges; CMS emits 1 wide edge. If both sides emitted at the same granularity, the byte-key 6-tuple comparison would resolve to 0 mismatches."*
   - **Correction:** CMS does not emit 1 wide edge in place of SHADOW's 4 per-byte; CMS emits 0 edges. Granularity alignment would not produce 0 mismatches — the gap is presence-vs-absence, not 4-vs-1.

8. **`6QIB_DESIGN.md:265-268`** (the "open architectural questions" stanza summarizing the residual) — same correction as 5 and 7: replace "4 vs 1 with different `intra_offset`" with "4 vs 0 because CMS scheduler reorders consumer before producer in NLL."

9. **`6QIB_DESIGN.md:286-287`** (the comparison table for byte-key 3-tuple vs 6-tuple) — Footnote needed: "the 6-tuple FAIL (30/90) is dominated by stream-position-ordering divergence in NLL, not edge-emission granularity at the `_resolve_producers` layer. See `n7og_PROBE_REPORT.md`."

In addition, the test file `Tensile/Tests/unit/test_n7og_edge_keys_multifixture.py:64` carries an inherited claim ("the SHADOW capture's `LoopBodyCapture.name_to_idx` is MISSING bindings for the rotating `ValuA/B_T0_I0` / `ValuA/B_X0_I0` pack-buffer registers") that this probe finds incorrect on the BPG#11 fixture as observed at the current branch tip — those names ARE bound, just to base value `-1` on the A side which still produces a sentinel-shaped byte-key for A-side reads. The 4% sentinel-edge fraction quoted upstream is consistent with sentinel being a minor contributor, not the dominant mechanism.

Updates needed to 6QIB_DESIGN.md.

---

## Probe addendum: cross-subiter ALU-producer exemption breakdown

**Probe script:** `/tmp/exemption_probe.py`.
**Full log:** `/tmp/exemption_probe.log`.
**Fixture:** `_BPG_11_TF32_4X4_TN` (canonical BPG#11, TF32 4×4 TN, `UsePLRPack=True`).
**Read-only.** No mutation of `CMSValidator.py` / `ScheduleCapture.py`. Probe replays each branch of `diagnose_missing_edge` over the 192 SHADOW-extra edges and counts which branch terminates.

### Branch counts

| Class | Branch (in `diagnose_missing_edge`) | File:line | Count |
|------|-------------------------------------|-----------|------:|
| **N1** | Legitimate-reorder defensive identity-equality fallback | `CMSValidator.py:3810-3817` | **0** |
| **N2** | Phase 0 `CaptureConsistencyError` (producer/consumer absent in subj) | `CMSValidator.py:3776-3783` | **0** |
| **N3** | Phase 1 cross-subiter ALU-producer exemption (returns `[]`) | `CMSValidator.py:3831-3843` | **192** |
| **N4** | Phase 1 `OrderInvertedFailure` | `CMSValidator.py:3844-3850` | **0** |
| **N5** | Phase 2 wait/barrier failures (MissingWait, WaitInsufficient, MissingBarrier, OverriddenInput, TimingTooClose) | `CMSValidator.py:3927-4023` | **0** |
| **N6** | `UnexplainedMissingEdgeError` fall-through | `CMSValidator.py:4043-4048` | **0** |
| | **Sum** | | **192** ✓ matches the 192 SHADOW extras |

The classification is **exhaustive and single-class**: every one of the 192 extras lands in exactly N3. Sum = 192 verified against `len(ref_keys - subj_keys)`.

### N3 detail (the only class that fires)

The 192 extras decompose as:

| dimension | breakdown |
|---|---|
| Producer category (ref/SHADOW) | `PackA0`: 96, `PackB0`: 96 |
| Consumer category (ref/SHADOW) | `MFMA`: 192 |
| Body pair (ref/SHADOW) | `(NLL, NLL)`: 192 |
| Subiter pair (REF, ref): `(p_subiter, c_subiter)` | `(0, 0)`: 96, `(0, 1)`: 48, `(0, 2)`: 48 |
| Edge kind | `raw_intrawave`: 192 |
| Intra-operand offsets observed | `(0,)`, `(1,)`, `(2,)`, `(3,)` (per-byte fan-out, 4 edges per consumer×operand×subiter combination) |

The exemption check is computed against the **subject (CMS) side** subiter values. On the CMS side, the same physical pack producers are tagged `PackA3` / `PackB3` (rotating-buffer subiter), so `p_node.subiter(nmps) == 3` and `c_node.subiter(nmps) ∈ {0, 1, 2}` — always non-equal, always satisfies `p_subiter != c_subiter`, always triggers the exemption. The reference (SHADOW) side carries `p_subiter = 0`, which is irrelevant to the check because the exemption uses subject-side bookkeeping.

### Representative example edges

**Example A** (intra=(3,), `PackA0 → MFMA`):

```
producer.canon = v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+11], v[vgprValuA_X0_I0+14], v[vgprValuA_X0_I0+15]
consumer.canon = v_mfma_f32_16x16x32_bf16 acc[4:7],
                 v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+3],
                 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[4:7]
edge_kind=raw_intrawave intra=(3,) src_slot=0 sink_slot=1
ref(SHADOW): p_body=NLL c_body=NLL p_pos=stream_index=30 c_pos=stream_index=55  p_sub=0 c_sub=0
subj(CMS):   p_body=NLL c_body=NLL p_pos=stream_index=128 c_pos=stream_index=14 p_sub=3 c_sub=0
```

On the subject side, the producer is at unrolled position 128, the consumer at 14 — producer-after-consumer in the body-local view. The producer is an ALU instruction (`VCvtPkF32toBF16`), `p_subiter=3 != c_subiter=0`, so the exemption fires and the edge is silently absorbed. Under the unrolled-program model (§5.1 of `6QIB_DESIGN.md`), the consumer at unrolled position 14 in NLL iter 0 would correctly resolve its read to the producer in the *previous* NLL iter (or the ML tail) that wrote the same byte-keys — no extra emitted, no exemption needed.

**Example B** (intra=(1,), `PackA0 → MFMA`, c_subiter=2):

```
producer.canon = v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+9], v[vgprValuA_T0_I0+6], v[vgprValuA_T0_I0+7]
consumer.canon = v_mfma_f32_16x16x32_bf16 acc[36:39],
                 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3],
                 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[36:39]
edge_kind=raw_intrawave intra=(1,) src_slot=0 sink_slot=1
ref(SHADOW): p_body=NLL c_body=NLL p_pos=stream_index=28  c_pos=stream_index=126 p_sub=0 c_sub=2
subj(CMS):   p_body=NLL c_body=NLL p_pos=stream_index=125 c_pos=stream_index=81  p_sub=3 c_sub=2
```

Same shape, different intra offset and a c_subiter=2 consumer. Note that here both the SHADOW and CMS schedules place the consumer *after* the producer in stream order on their own side — the validator still flags it as a missing edge because the canonical-render strings differ. Under the unrolled walk this collapses (the byte-key resolution is identical on both sides); the exemption is doing the same silencing work the unrolled walk would do for free.

### Synthesis

- Phase 0 (N2) catches **0** of the BPG#11 extras. The identity-set coverage gate at `compare_graphs` entry is not the relevant filter for these extras; both producer and consumer identities are present in both graphs (40 of 64 CVT instances in NLL share `id(rocisa_inst)` between SHADOW and CMS per the upstream probe, and the identity tuple `(canonical_render, emission_ordinal)` is identical for the shared instances).
- The cross-subiter ALU-producer exemption (N3) absorbs **100%** of BPG#11's residual divergence (192/192). It is the only branch that fires for SHADOW extras on this fixture. The exemption is what is currently "hiding" the validator's body-local `latest_writer` blind spot on BPG#11.
- No `UnexplainedMissingEdgeError` (N6) occurs. The exemption is comprehensive enough to keep BPG#11 quiet, which is why the underlying blind spot has remained undiagnosed: removing the exemption without first landing the unrolled walk (§5.1 in `6QIB_DESIGN.md`) would convert 192 silent `[]` returns into 192 `OrderInvertedFailure` findings — none of which represent real defects — making the validator unusable on `UsePLRPack=True` fixtures.
- The most-surprising finding: the exemption isn't a niche carve-out catching the occasional false positive; on BPG#11 it is the sole non-trivial branch in the entire SHADOW-extra classification. The validator's apparent "green" status on this fixture is entirely produced by the exemption silently returning `[]`. Removing it surfaces the real story; replacing the body-local walk with the unrolled-program walk eliminates the need for it.

### How to reproduce

```bash
WT=/home/alvasile/rocm-libraries/.worktrees/validator_long_term_plans/projects/hipblaslt/tensilelite
cd $WT
PYTHONPATH=$WT /home/alvasile/venv/bin/python3 /tmp/exemption_probe.py 2>&1 | tee /tmp/exemption_probe.log
```

The probe takes ~30s to build the SHADOW/CMS pair. Output prints the N1–N6 counts and the first two example edges per class.
