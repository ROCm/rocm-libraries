# Design — `rocm-libraries-6qib`: edge_keys must be both allocation-invariant AND order-sensitive

**Status:** decision-required (2026-06-02)
**Companion bead:** `rocm-libraries-6qib` (P0; blocks `r62g` Phase 3, `udqg`, `32tg`)
**Supersedes:** `udqg` (mechanism wrong), `32tg` (3-tuple match insufficient)
**Empirical foundation:** `Tensile/Tests/unit/test_n7og_edge_keys_multifixture.py` + `n7og_CORRECTNESS_REPORT.md`

---

## 0. Primer — the vocabulary this doc uses

If you haven't read the validator design docs, the terms below get used heavily.

### 0.1 A dataflow edge

An **edge** in this validator's `DataflowGraph` represents one *register-or-memory dataflow dependency*: producer instruction X wrote some bytes, consumer instruction Y read some of those same bytes, so Y depends on X. Each edge carries:

- A producer node (a captured instruction)
- A consumer node (a captured instruction)
- An `edge_kind` (the dependency type — `raw_intrawave`, `lds_raw_intrawave`, etc.)
- `intra_operand_byte_offset` — see §0.3
- `src_operand_slot` / `sink_operand_slot` — see §0.3

### 0.2 Byte keys

Every register or memory write covers a set of bytes. `_byte_keys_for_resource` (`ScheduleCapture.py:1428`) converts a resource (register range, memory slice) to a tuple of byte-grain identifiers — **one entry per byte the operand covers.**

| Resource | byte_keys |
|---|---|
| `v[15]` (1 VGPR) | `(('v', 15),)` — one key |
| `v[12:15]` (4 VGPRs) | `(('v', 12), ('v', 13), ('v', 14), ('v', 15))` — four keys |
| `s[8:9]` (2 SGPRs) | `(('s', 8), ('s', 9))` — two keys |
| `vgprValuA_X0_I0+12` (symbolic, with name_to_idx resolving to base 12) | `(('v', 12),)` — resolved to numeric |
| `vgprFoo` (symbolic, no name_to_idx entry) | `(('v', 'vgprFoo', 0),)` — symbolic key |
| LDS slice `[offset=64, byte_count=16]` | `(('mem', 'lds', buf, 64), ..., ('mem', 'lds', buf, 79))` — 16 mem keys |

Byte keys are **allocation-invariant for numeric registers**: `v[12]` is `('v', 12)` regardless of which symbolic name the writer used. That's the whole point of byte-keying — it strips register-allocation noise.

### 0.3 Per-byte vs wide edge — what changes when producer and consumer touch multiple bytes

When the producer writes N bytes and the consumer reads N overlapping bytes, the edge between them can be *materialized* two ways:

**Wide edge** (1 edge total): one `DataflowEdge` with `intra_operand_byte_offset=(0,1,2,3)` — meaning "this single edge covers bytes 0..3 of the read operand."

**Per-byte edges** (N edges total): four `DataflowEdge`s, each with `intra_operand_byte_offset=(0,)`, `(1,)`, `(2,)`, `(3,)` — meaning "this edge covers byte 0 of the read operand," then "byte 1," etc.

Same physical dataflow, two representations. The choice is made in `_resolve_producers` (`ScheduleCapture.py:1482`): bytes get grouped by `(writer_node, write_resource, write_slot)`. If both 4 bytes come from the same writer + same write resource + same write slot → 1 wide edge. If they came from different writers, or the same writer's different slots → multiple edges.

**Earlier drafts of this doc speculated that SHADOW and CMS differ on this representation choice. The probe at `n7og_PROBE_REPORT.md` empirically disproved that on BPG#11.** The actual source of edge-count divergence is documented in §0.7 below; per-byte vs wide-edge is correctly identical between SHADOW and CMS captures for the same physical instructions.

### 0.4 `intra_operand_byte_offset`

The `intra_operand_byte_offset` field is a tuple of integers, each an index INTO THE CONSUMER'S READ OPERAND (0..N-1). It says "this edge represents bytes at these positions within the consumer's read." It is allocation-invariant by construction — these are positions, not register names.

Example: consumer is `v_mfma ..., v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+15]` (a 4-byte read). One producer wrote `v[12:13]` (bytes 0-1 of the read) and another wrote `v[14:15]` (bytes 2-3). Two edges:
- Edge 1: producer A, `intra_operand_byte_offset=(0, 1)`
- Edge 2: producer B, `intra_operand_byte_offset=(2, 3)`

### 0.5 `src_operand_slot` / `sink_operand_slot`

Positional integer indices (0, 1, 2, ...) saying WHICH operand of the producer was written and WHICH operand of the consumer was read. E.g., for `v_swap_b32 vA, vB` both `vA` and `vB` are written AND read; src_slot/sink_slot disambiguate which positional slot the edge is referring to.

### 0.6 idMap category names (`PackB0`, `PackA1`, etc.)

The CMS idMap (`build_idmap` at `ScheduleCapture.py:1040`) labels instructions by category. `PackA{u}` and `PackB{u}` are the per-subiter "pack code" buckets for the A and B operands of the GEMM (`u` is the unroll-loop subiter index, 0..numLoopIter-1). Other relevant categories: `LRA{u}`/`LRB{u}` (local-read per-iter), `MFMA` (the matrix instruction), `LWA`/`LWB` (local-write), `GRA`/`GRB` (global-read), `LCC` (loop counter code).

When this doc says "PackB0 vs PackB3," it means one side captured an edge under the `PackB0` category (subiter 0) and the other captured it under `PackB3` (subiter 3). That's a per-iter scope mismatch — same physical instruction, different bookkeeping.

### 0.7 Body-local `latest_writer` walk — the true source of NLL/NGL divergences

`build_dataflow_graph` processes nodes in stream-position-sorted order WITHIN EACH BODY (`PRO`, `ML-1`, `ML`, `NGL`, `NLL`). It maintains a `latest_writer[byte_key]` dict that gets **initialized empty at the start of each body** and updated by writes as the walk progresses. Reads query `latest_writer[bk]` and attach edges only to writers seen earlier in the same body's stream walk.

**Implication:** if a body's first MFMA consumer reads a VGPR that was written in the PREVIOUS body (e.g. ML's tail) — i.e. a cross-body / cross-iter live-in — the validator sees an empty `latest_writer[bk]` and emits ZERO edges into that consumer. The dataflow IS there in the kernel; the validator just doesn't model it.

**`UsePLRPack=True` is the specific feature that triggers this pattern.** The Tensile scheduler flag (`Tensile/Common/ValidParameters.py:959`) enables a software-pipelined rotating pack-buffer convention: subiter 3's pack code prepares the values that subiter 0's MFMA in the **next** loop iteration will consume. This is the meaning of the T0/X0 register-naming alternation in the operands (`vgprValuA_T0_I0` and `vgprValuA_X0_I0` are two halves of the rotating buffer). The scheduler explicitly halves the pack-buffer VGPR allocation under this mode (`KernelWriter.py:8821`, `:8861`) because the buffer is reused across iters.

The bf16 negative pin in `test_n7og_edge_keys_multifixture.py` (`UsePLRPack=False`) shows **0 mismatches** on both 6-tuple byte-key and identity-tuple bases — direct empirical confirmation that the flag is the trigger.

When SHADOW and CMS schedule the same NLL physical instructions in different stream orders (both valid given the cross-iter live-in semantics), the body-local `latest_writer` walk produces different edge sets — even though the two schedules are dataflow-equivalent at the kernel level. The residuals are not bugs in either schedule; they are blind spots in the validator's body-local dataflow model.

**Why SHADOW and CMS produce different orders even with the same `kernel["UsePLRPack"]=True`.** The kernel-level flag is observed by both pipelines, but the SCHEDULER-level pipelining optimization is deliberately gated off when CMS is in charge. At `Tensile/KernelWriter.py:9066`:

```python
# do prefetch and scheduling for full pack code
# this sceduling opt is for non CMS. No need to enable it for CMS
self.states.doFullPackCodePrefetch = kernel["UsePLRPack"] and not kernel["UseCustomMainLoopSchedule"]
```

For BPG#11 (`UsePLRPack=True` AND `UseCustomMainLoopSchedule=1`), `doFullPackCodePrefetch=False`. The default scheduler (SHADOW source) sees the rotating-buffer naming and halved VGPR allocation (those parts are universal) but does NOT do the cross-iter prefetch ordering — it emits packs in the linear baseline order (producer-first within each body). The CMS scheduler implements its own cross-iter pipelining in `dispatch.py` separately.

So under CMS=1 the two schedules emit the **same physical rocisa instructions** (per the probe: 40 of 64 NLL CVT instances share `id(rocisa_inst)` directly between the captures — same Python objects) but in **different stream orderings**:
- SHADOW: producer-first linear (default scheduler's UNoptimized-under-CMS baseline)
- CMS: consumer-first pipelined (CMS's hand-tuned cross-iter schedule)

This is the EXPECTED contract for SHADOW vs CMS captures per design v5 §1 — they should differ ONLY in scheduling, not in dataflow. The fact that today the validator surfaces this ordering difference as edge-count divergence (208 mismatches on BPG#11) is the body-local-walk blind spot — not a bug in either schedule.

### 0.8 Why the existing cross-subiter ALU-producer exemption is an anti-pattern

`CMSValidator.py:3831-3843` carries a Phase-1 carve-out inside `diagnose_missing_edge`:

```python
if default_p_before_c and not subj_p_before_c:
    # Cross-subiter ALU-producer ... "legitimate pipelining" ...
    nmps = subj_graph.num_mfma_per_subiter
    if (_is_alu_producer(p_node)
            and p_node.subiter(nmps) != c_node.subiter(nmps)):
        return []  # cross-subiter pipelined dependency — legitimate
```

The exemption fires whenever (i) the default ordering puts producer before consumer, (ii) the subject ordering puts the producer after the consumer in the same body, (iii) the subject producer is an ALU instruction, and (iv) the subject-side subiter labels of producer and consumer differ. On a match it silently returns `[]` — no failure, no diagnostic, no edge recorded.

**It pattern-matches the SYMPTOM, not the dataflow.** The conditions above are exactly the surface shape of a cross-subiter rotating pack-buffer pipelining handoff under `UsePLRPack=True` (subiter 3's pack writes the registers subiter 0's MFMA in the next iteration consumes). But the check does NOT verify that:

- The producer's value is actually what the consumer reads (it never inspects byte-keys or `latest_writer` state across iters).
- No other writer displaced that value between iters (an intervening clobber would silently pass).
- The producer-consumer pair are actually iteration neighbors in the rotating-buffer sense (any same-body subiter mismatch satisfies the check, even ones with no dataflow relation).

It is a syntactic guard that returns the right answer for the legitimate `UsePLRPack` pipelining case and the wrong answer (silence) for any failure mode that happens to share the same surface shape — a real cross-subiter reorder, a clobber, a missing wait that happened to land in this dispatch branch. The validator is exchanging coverage for green tests on a single fixture.

**It papers over a representational problem.** The fundamental issue is that `build_dataflow_graph`'s body-local `latest_writer` walk (§0.7) cannot resolve cross-iter live-ins. The principled fix is to model the dataflow across iter boundaries directly — concatenate the body streams in execution order and do a single linear walk (see §5 (d) below). With that model, the legitimate cross-subiter pack producer is the most-recent prior writer the consumer reads from; the edge resolves naturally, the comparison emits no spurious extra, no exemption needs to fire, and any *real* reorder or clobber still surfaces because the dataflow model now distinguishes them on data grounds rather than on category-label heuristics.

**Empirical evidence (exemption-classification probe on BPG#11):** all 192 SHADOW-extra edges land in this exemption branch (N3 = 192). Phase 0 (N2), legitimate-reorder defensive identity fallback (N1), Phase-1 `OrderInvertedFailure` (N4), Phase-2 wait/barrier coverage (N5), and `UnexplainedMissingEdgeError` (N6) all score zero. The exemption is the *only* thing stopping these from being routed as `OrderInvertedFailure` findings. Producer-category breakdown: 96 PackA0 + 96 PackB0 — exactly the pack-MFMA edges. Subiter pair breakdown (reference side / subject side): producers always at ref-subiter 0 and subj-subiter 3, consumers at subj-subiter ∈ {0, 1, 2} — the rotating-buffer pattern unmistakably. The exemption is currently absorbing 100% of BPG#11's residual divergence, which (in the user's mental unrolled-loop model) shouldn't be divergence at all.

The doc treats the exemption as load-bearing infrastructure throughout earlier sections; it is in fact a workaround for the body-local walk's blind spot and should be removed once that blind spot is closed (§5 (d)).

### 0.9 Per-body breakdown — where the divergence actually lives

The 208 SHADOW-vs-CMS edge_keys mismatches on BPG#11 are not uniformly distributed across the kernel's bodies; they are concentrated in the tail-loop bodies where cross-iter live-ins occur. Per-body table from the probe:

| Body | SHADOW nodes | CMS nodes | SHADOW edges | CMS edges | Δ edges | Notes |
|------|-------------:|----------:|-------------:|----------:|--------:|-------|
| PRO   | 0   | 0   | 0   | 0   | — | Empty in this fixture |
| ML-1  | 0   | 0   | 0   | 0   | — | Empty in this fixture |
| **ML**    | 20  | 20  | **17**  | **17**  | **0** | Fully aligned — body-local walk works for the steady state |
| **NGL**   | 28  | 28  | **78**  | **94**  | **+16 missing-in-SHADOW** | CMS has 16 edges SHADOW doesn't |
| **NLL**   | 136 | 136 | **552** | **360** | **+192 extra-in-SHADOW** | SHADOW has 192 edges CMS doesn't |

Observations:

- **ML is clean** because each ML iteration's pack→MFMA dependency lives within a single ML iter window. The body-local walk's `latest_writer` map is populated and queried within the same conceptual iteration; no cross-iter live-in to model.

- **NLL is the dominant failure body** (192 of 208 total mismatches). NLL is the no-load-loop tail — the unrolled tail iterations after global-load issuance stops. NLL's first MFMA consumers read values written by the previous mainloop iteration's pack code (back in ML); NLL's CVT producers prepare values for the *next* iter's MFMAs. The body-local walk sees consumers at NLL stream-index 0 with empty `latest_writer[bk]`, so emits 0 edges from CMS into them; SHADOW's UNoptimized ordering puts the producers first, so SHADOW emits 192 edges. All 192 are absorbed by the exemption today (§0.8).

- **NGL has a 16-edge asymmetry in the OPPOSITE direction** — these are edges CMS has but SHADOW doesn't (`LR3→MFMA` in NGL, 8 edges per operand × 2 operands per the probe).

**The asymmetric-direction finding has a load-bearing consequence: `compare_graphs` only checks one direction.** At `CMSValidator.py:3711`:

```python
ref_keys = reference.edge_keys()
subj_keys = subject.edge_keys()
missing_keys = ref_keys - subj_keys   # SHADOW − CMS
```

Only edges that are in the reference (SHADOW) but not in the subject (CMS) get routed through `diagnose_missing_edge`. So the **192 NLL extras** (`ref − subj`) ARE processed by `compare_graphs` (and silently absorbed by the exemption); the **16 NGL missing-in-SHADOW** (`subj − ref`) are NEVER processed — they never reach the classifier at all. The build-time inline xj16 assertion is completely blind to them.

The probe test `test_n7og_edge_keys_multifixture.py` surfaces both directions because it does its own symmetric set-diff (`default_edges - cms_edges` AND `cms_edges - default_edges`) — that's why the test reports 208 total while compare_graphs only sees 192 candidate failures.

**Implication for (d):** the unrolled-program walk must close BOTH directions. The acceptance criterion in §5 (d) needs to assert symmetric edge equality on BPG#11 — both NLL's 192 and NGL's 16 resolve to 0 — not just check that compare_graphs returns empty (which is satisfied today, vacuously, on the unprocessed half).

**Implication for the validator's correctness contract:** the current one-direction-only check in `compare_graphs` is itself a gap — subject-extras that the reference doesn't have are exactly the shape of a CMS schedule emitting a write the default didn't, which is a real defect class. This may warrant its own bead (file P0 with `br dep add r62g` if so) — but that's a separate concern from the body-local-walk blind spot and shouldn't be conflated.

---

## 1. Problem statement

### 1.1 What `edge_keys()` does today

`DataflowGraph.edge_keys()` at `Tensile/Components/CMSValidator.py:1300` returns a `set` of 6-tuples — one tuple per dataflow edge in the graph. Each tuple is:

```python
(producer.identity, consumer.identity,
 edge_kind, intra_operand_byte_offset,
 src_operand_slot, sink_operand_slot)
```

Where:
- `producer.identity` and `consumer.identity` are each `(canonical_render, emission_ordinal)`.
- `canonical_render` is the literal rendered assembly text of the instruction, **including register operand names** like `v[vgprValuA_T0_I0+0]`. Two instructions with different operand names produce different `canonical_render` strings → different identities → different edge_keys.
- `emission_ordinal` is a per-body sequence counter (`assign_emission_ordinals` at `ScheduleCapture.py:754`). Same text emitted at two different positions in the stream produces two different ordinals.
- `edge_kind` / `intra_operand_byte_offset` / `src_operand_slot` / `sink_operand_slot` are per §0.

`compare_graphs()` (`CMSValidator.py:3735`) consumes two such sets — one from the reference graph, one from the subject graph — and does set-difference both ways. Any tuple in reference but not subject is a "missing edge"; any tuple in subject but not reference is an "extra edge." `diagnose_missing_edge` (`CMSValidator.py:~3680`) then classifies each missing edge (e.g. as `OrderInvertedFailure`, `UnexplainedMissingEdgeError`, etc.).

### 1.2 The two contradictory requirements

| Property | Why it's needed | Lost if we drop |
|---|---|---|
| **Allocation-invariant** | Register-allocator-induced renaming between SHADOW and CMS captures should not flag false positives. Two schedules that emit the same logical edge against different physical VGPRs (e.g. `vgprValuA_T0_I0+0` vs `vgprValuA_X0_I0+12`) are semantically equivalent. | n7og fixture: false-positive cascade (208 mismatches on BPG#11, 624 on oplb-style) |
| **Order-sensitive** | If the CMS scheduler reorders two instructions such that a producer now follows its consumer, the edge "consumer reads X" before "producer writes X" is a real defect (an `OrderInvertedFailure`). The order of the producer/consumer in the emission stream MUST be encoded in the edge key, or the set-difference comparison cannot detect it. | 11 reorder/SCC/carveout tests fail: `test_pack_before_swap_orderinverted`, `test_real_kernel_neutralized_carveout_surfaces_768_pack3_mfma_failures`, the 5 `test_ValidateSCCoverlap.py::*`, etc. |

The current identity-tuple basis satisfies (2) trivially (canonical_render embeds operand names → strict equality acts as both identity AND a position marker because positions in the rendered stream produce unique `emission_ordinal`s). It violates (1) — that's the false positives.

A pure byte-key basis satisfies (1) trivially (byte-keys are register-allocation-independent). It violates (2) — different emission orders of the same physical-byte flow produce the same edge_keys, so reorder is undetectable.

---

## 2. Concrete examples from real schedules

### 2.1 BPG#11 (TF32 TN MT 128x160x64, `UsePLRPack=True`)

**Fixture:** `Tensile/Components/CustomSchedule/gfx950/test_yamls/6hk3_tf32_128x160x64_tn.yaml`

**Empirical setup:** SHADOW vs CMS captures harvested via `real_kernel_capture_pair` fixture, both go through `build_dataflow_graph()` and `edge_keys()` is set-diffed.

#### Under current identity-tuple basis (Approach 0 / status quo)
- **208 mismatches** (16 missing in SHADOW / 192 extra in SHADOW), all in body `ML`
- Both sides emit the same logical pack-MFMA chain
- The diff is dominated by edges whose endpoints render to **different register operand names** in the captured text:

| Side | A sample edge in `(producer.canonical_render, consumer.canonical_render, ...)` |
|---|---|
| SHADOW | `(v_cvt_pk_bf16_f32 v[vgprValuA_T0_I0+0], ..., v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+0:vgprValuA_T0_I0+0+1], ..., PackA0→MFMA, ...)` |
| CMS | `(v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+12], ..., v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+1], ..., PackA0→MFMA, ...)` |

Both edges encode "PackA0 emits a converted bf16 value → MFMA consumes it as src0." Under set-diff each looks like an "edge present on one side that isn't on the other." 208 false positives.

#### Under pure byte-key basis (Approach E, what 32tg recommended)

Replace `producer.identity` / `consumer.identity` in the edge key with the **byte keys** of the producer's write resource and the consumer's read resource (see §0.2). This makes the edge key allocation-invariant — `vgprValuA_T0_I0+0` and `vgprValuA_X0_I0+12` both resolve to `('v', 12)` byte keys (assuming both name_to_idx tables agree on the base, which they do here).

Two measurement variants tried:

- **3-tuple `(prod_byte_key, cons_byte_key, edge_kind)`:** **0 mismatches** (139 unique edges on each side, match exactly). But the 3-tuple drops `intra_operand_byte_offset` + `src_operand_slot` + `sink_operand_slot`, which the design contract says are part of the comparison.
- **6-tuple `(prod_byte_key, cons_byte_key, edge_kind, intra_offset, src_slot, sink_slot)`:** **30 mismatches remain on BPG#11; 90 on the oplb-style fixture.** Originally hypothesized to be per-byte-vs-wide-edge asymmetry; empirical probe (`n7og_PROBE_REPORT.md`) disproved that. Real mechanism in §0.7 above and worked out below.

**Concrete worked example of one of the 192 residuals (when measured at the full identity-tuple basis; the 30 from the byte-key 6-tuple are a subset):**

The failing body is `NLL`, not `ML`. ML is fully aligned (17 edges on both sides). All 192 extras live in NLL. Empirical trace:

```
SHADOW NLL stream:
  stream_index= 7  PackB0  v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+0], ...   # writes ('v', 31)
  stream_index= 8  PackB0  v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+1], ...   # writes ('v', 32)
  stream_index= 9  PackB0  v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+2], ...   # writes ('v', 33)
  stream_index=10  PackB0  v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+3], ...   # writes ('v', 34)
  stream_index=14  MFMA    v_mfma_f32_16x16x32_bf16 acc[0:3],
                           v[vgprValuB_X0_I0+0:+3], ...                   # reads ('v', 31..34) — producers visible

CMS NLL stream (SAME rocisa instances, different schedule):
  stream_index= 0  MFMA    v_mfma_f32_16x16x32_bf16 acc[0:3],
                           v[vgprValuB_X0_I0+0:+3], ...                   # reads ('v', 31..34) — latest_writer empty
  stream_index= 3  MFMA    ... (more MFMA reads)
  stream_index= 7  MFMA    ...
  stream_index=84  PackB3  v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+0], ...   # writes ('v', 31) — AFTER its supposed consumer
  stream_index=85  PackB3  v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+1], ...   # writes ('v', 32)
  stream_index=87  PackB3  v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+2], ...   # writes ('v', 33)
  stream_index=88  PackB3  v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+3], ...   # writes ('v', 34)
```

| Side | Edges emitted into the MFMA consumer | Mechanism |
|---|---|---|
| **SHADOW** | 4 narrow edges, one per byte position:<br>`(prod_bk, cons_bk, raw_intrawave, (0,), 0, 0)`<br>`(prod_bk, cons_bk, raw_intrawave, (1,), 0, 0)`<br>`(prod_bk, cons_bk, raw_intrawave, (2,), 0, 0)`<br>`(prod_bk, cons_bk, raw_intrawave, (3,), 0, 0)` | At consumer's `stream_index=14`, `latest_writer[('v', 31..34)]` was populated by the four `VCvtPkF32toBF16` writes at stream_index 7-10. Each byte resolves to a distinct writer (4 different physical CVTs writing 4 different physical VGPRs), so `_resolve_producers` correctly emits 4 narrow edges. |
| **CMS** | **0 edges.** | At consumer's `stream_index=0`, `latest_writer[('v', 31..34)]` is empty — no writer has been seen yet in this body. The same four `VCvtPkF32toBF16` instances exist in the CMS NLL stream but at positions 84-88, AFTER the consumer. `_resolve_producers` returns no producers → 0 edges. |

**The four CVT writers are not duplicates and not wide-vs-narrow alternatives.** They are four genuinely-different physical instructions each writing a different VGPR (`+0`, `+1`, `+2`, `+3`). 40 of the 64 CVT instances in NLL are bit-equal Python objects (same `id(rocisa_inst)`) between SHADOW and CMS — confirming both sides see the SAME physical rocisa-IR. The divergence is purely positional.

Under set-diff with the 6-tuple key (or identity-tuple key), each of SHADOW's 4 narrow tuples for this consumer is "extra" because CMS has no edges to match against. Multiplied by 48 MFMA consumers × 4 bytes × 2 operands across NLL gives the 192 figure on BPG#11.

**This is not a capture-layer granularity divergence.** It is the body-local `latest_writer` walk (§0.7) failing to model the `UsePLRPack=True` rotating pack-buffer semantics. Under `UsePLRPack`, subiter 3's pack code (`PackB3` in CMS's categorization) writes the rotating-buffer half that subiter 0's MFMA in the NEXT iteration reads. So a CVT at NLL stream_index 84 categorized `PackB3` is correctly placed AFTER an MFMA consumer at stream_index 0 — they belong to different conceptual iterations. The kernel is correct; the validator's body-local walk just doesn't model the rotating pack-buffer pipelining.

The bf16 negative pin in `test_n7og_edge_keys_multifixture.py` (`UsePLRPack=False`) shows 0 mismatches — direct empirical confirmation that disabling the pipelining eliminates the symptom.

No edge-key tuple shape on its own solves this. Three angles for the fix:
1. **Seed `latest_writer` at body boundaries** from the previous body's tail state, so cross-iter live-ins resolve correctly. Touches `build_dataflow_graph` Phase 2, not `_resolve_producers` or `edge_keys`. Right architectural fix; out of scope for the immediate Phase 3 unblock.
2. **Classify the resulting deltas as `OrderInvertedFailure`** via a compound-key `ordinal_class` (Approach (a) below) so they surface as a known class of validator finding rather than silently inflating false positives. Doesn't fix the underlying blind spot but makes it diagnosable.
3. **Filter NLL/NGL out of cross-graph comparison** when the body has an empty `latest_writer` at first-consumer-resolution time. Tactical workaround; not recommended.

#### Tests that BREAK under pure byte-key
- `test_validate_pack_graph.py::test_pack_before_swap_orderinverted` — pins exactly the reorder case: producer emitted AFTER consumer should fail. Under byte-key, allocation-invariant set membership ignores order. Edge_keys match → set-diff returns empty → `diagnose_missing_edge` is never invoked → `OrderInvertedFailure` Phase 1 detection never runs.
- `test_ValidateSCCoverlap.py::*` (5 tests) — same shape for SCC-clobber detection
- `test_cross_subiter_alu_carveout_real_kernel.py::test_real_kernel_neutralized_carveout_surfaces_768_pack3_mfma_failures` — exactly the test name says what it pins
- `test_cross_subiter_pack_artifact.py::TestCrossSubiterPackArtifact::test_carveout_suppresses_artifact_and_neutralization_surfaces_it`
- `test_validate_gr_not_too_early_graph.py::TestGRNotTooEarlyDtlPlusLdsBufGraph::test_negative_one_prev_iter_lr0_not_drained`

### 2.2 oplb-style (TF32 TN MT 192x256x32, `UsePLRPack=True`)

- 624 mismatches under identity-tuple (48 missing / 576 extra)
- 0 under 3-tuple byte-key
- 90 remaining under 6-tuple byte-key (same body-local-`latest_writer` stream-ordering shape as BPG#11, scaled — see §0.7 and §2.1's worked example; this is NOT a granularity asymmetry)

### 2.3 bf16 (256x256x64 TN, `UsePLRPack=False`) — positive pin

- 0 mismatches under either identity-tuple OR byte-key
- Establishes that without rotating pack-buffer, there's no false-positive cascade
- Serves as a regression pin: any future change that adds new false positives on this fixture must be investigated

### 2.4 Reorder example (synthetic, from `test_pack_before_swap_orderinverted`)

The test constructs a small graph where:
- Producer: `v_pack_b32 v10, v8, v9`
- Consumer: `v_swap_b32 v10, v11`
- Both on the same physical VGPR `v10`

Subject schedule emits them in the WRONG order (consumer before producer). Reference emits in the right order. Identity-tuple basis: subject's edge_key has `emission_ordinal=N+M` for the consumer-before-producer arrangement; reference has `N` — different keys, set-diff catches it, `OrderInvertedFailure` Phase 1 fires. Byte-key basis: both have key `(('v', 10), ('v', 10), 'pack→swap', 0, 0, 0)` — identical, set-diff empty, no failure surfaced.

---

## 3. Proposed approaches

### 3.1 Approach (a) — Compound key: byte-key + emission ordinal

**Shape:** `(prod_byte_key, cons_byte_key, edge_kind, intra_offset, src_slot, sink_slot, ordinal_class)`

where `ordinal_class` is a discrete partition of the edge's position in the emission stream — e.g.
- "producer-before-consumer" / "consumer-before-producer" (binary)
- Or `(emission_ordinal of producer, emission_ordinal of consumer)` pair

**How it resolves both requirements:**
- (1) Allocation-invariant: byte-keys replace register-name embedding; register-renaming differences vanish
- (2) Order-sensitive: ordinal_class makes consumer-before-producer produce a different key than producer-before-consumer

**Implementation cost (estimated):**
- `edge_keys()` change: ~20-40 lines
- Update `diagnose_missing_edge` to thread ordinal_class through Phase 1 detection
- Possible test fixture updates (some `OrderInvertedFailure` tests may currently assert specific keys)
- BPG#11 and oplb fixtures should resolve to 0 mismatches (allocation-invariant)
- 11 reorder/SCC/carveout tests should preserve their detection behavior (order-sensitive)

**Risks:**
- The 30+90 6-tuple residuals (§2.1) are SHADOW-emits-N-narrow vs CMS-emits-0 deltas caused by the body-local `latest_writer` walk's blind spot for cross-iter live-ins in NLL/NGL (§0.7). Compound key + `ordinal_class` does NOT fix the underlying blind spot, but it DOES change the routing: the "extra in SHADOW" tuples will have an `ordinal_class` indicating producer-before-consumer, which is the legitimate stream order — they then surface through `diagnose_missing_edge` as `OrderInvertedFailure`-class findings rather than silently inflating false-positive counts. This is improvement on the current state but not closure; the principled fix is body-boundary `latest_writer` seeding, tracked separately.
- Choosing what `ordinal_class` is at the right granularity. Too fine (raw ordinals) → false positives reappear when both sides reorder symmetrically. Too coarse (binary) → may miss multi-instruction reorder cycles.

**Verdict:** Most directly addresses the dual requirement. Lowest implementation risk. Surfaces the NLL/NGL residuals through a known classification path (good) but does not close the body-boundary blind spot.

### 3.2 Approach (b) — Capture-layer alignment

**Shape:** Don't change `edge_keys()`. Instead, change how SHADOW emits edges so it matches CMS's granularity (or vice versa).

Originally proposed against the (mistaken) belief that the 6-tuple residuals were "SHADOW emits 4 per-byte vs CMS emits 1 wide." The probe at `n7og_PROBE_REPORT.md` empirically refuted that: CMS emits ZERO edges in the residual cases, not one wide edge. There is no granularity to align — the gap is presence-vs-absence in `latest_writer`, not coalescing-vs-not.

**How it would have resolved both requirements (per the original framing):**
- (1) Indirect — if both sides emit identically, allocation differences also align (in principle)
- (2) Still using identity-tuple basis, so order is preserved trivially

**Why it does not actually fix the BPG#11 / oplb residuals:** the residuals are not "SHADOW per-byte vs CMS wide-edge" — they are "SHADOW per-byte vs CMS none-at-all" caused by the stream-order placement of consumers before producers in CMS NLL. There is no granularity to align between. Aligning `_resolve_producers` granularity changes neither side's `latest_writer` state at consumer-resolution time.

**The actual capture-layer fix that WOULD address the NLL/NGL residuals** is body-boundary `latest_writer` seeding (extending `build_dataflow_graph` Phase 2 to carry forward the previous body's tail `latest_writer` state into the next body's start, so cross-iter live-ins resolve to their previous-body producers). This is a distinct, larger refactor — file as a separate P0 bead with `br dep add r62g <bead>` rather than rolling into 6qib.

**Verdict:** Original (b) framing (granularity coalescing) is moot per the probe. The closest principled replacement is body-boundary `latest_writer` seeding, which is a separate scope and should be its own bead.

### 3.3 Approach (c) — Two-phase comparison

**Shape:** Keep identity-tuple edge_keys. Add a pre-filter that strips known false-positive classes (register-naming under UsePLRPack) before the set-diff.

**How it resolves both requirements:**
- (1) Filter step removes register-renaming false positives explicitly
- (2) Identity-tuple set-diff preserves order detection on what's left

**Implementation cost:**
- Define what counts as a "register-renaming false positive" — by-pattern matching on canonical_render
- Add filter step before set-diff
- ~50-100 lines

**Risks:**
- The filter is a new enumeration — same shape as the m7o5/nmsx scrub-list anti-pattern. Adding a new register-rename pattern (next codegen change) silently re-introduces false positives.
- Dual semantics: comparison happens in two modes. Harder to reason about.
- Per the user's standing rule: "if filing a new bead is needed, make it required as soon as possible" — this approach IS a workaround that defers the principled fix.

**Verdict:** Quick to implement, but introduces the centralized-list anti-pattern. Not recommended.

---

## 4. Additional considerations

### 4.1 What the empirical investigation revealed

The investigation chain (n7og → udqg → 32tg → 6qib) progressively narrowed the understanding:

1. **n7og** — speculative: "identity-tuple embeds register names; might cause oplb-style false positives." Filed P0.
2. **udqg** — first investigation claimed "SHADOW name_to_idx missing pack-buffer bindings → sentinel byte-keys." **Empirically wrong** (correctness verifier showed bindings ARE present, only 4% of mismatches are sentinel-related).
3. **32tg** — correctness verifier's correction: "Approach E (byte-key 3-tuple) DOES fix the n7og fixture." **Mostly right but incomplete** — only proved on 3-tuple basis; with the full 6-tuple basis from the existing contract, 30 mismatches remained.
4. **6qib** (this doc) — fix attempt proved (a) 6-tuple byte-key still has structural residuals AND (b) byte-key matching breaks 11 reorder/SCC/carveout tests. **Both pieces of evidence required to see the real dual-requirement conflict.**

Each round was a real refinement, not noise. The earlier framings weren't fully wrong — they were correct about parts of the problem.

### 4.2 What's NOT addressed by any of these approaches

**Structural edge-granularity divergence between SHADOW and CMS** (the 30 / 90 6-tuple residuals on BPG#11 / oplb):

SHADOW: 4 edges with `intra=(0,)/(1,)/(2,)/(3,)`, category `PackB0`
CMS:    1 edge with `intra=(0,1,2,3)`, category `PackB3`

This is a separate problem from the dual-requirement conflict. Approach (a) makes them DIAGNOSABLE by routing them through `OrderInvertedFailure` classification (instead of silently inflating false-positive counts) but does NOT fix the underlying body-boundary `latest_writer` blind spot. Approach (b) was originally proposed to align granularity — moot per the probe (no granularity to align). The principled closure is body-boundary `latest_writer` seeding, which is a separate scope.

If we land Approach (a) and the n7og fixtures still show 30+90 mismatches (now classified as `OrderInvertedFailure`-class findings rather than raw extras), those should be a NEW P0 bead for body-boundary `latest_writer` seeding with `br dep add r62g <new-bead>`. The standing no-deferred-discoveries rule applies.

### 4.3 Why fixing this matters for Phase 3

`rocm-libraries-r62g` (Phase 3) is the hard go/no-go gate against the CMS test surface. The dataflow-graph layer's `compare_graphs()` is at the center of that gate. If `edge_keys()` produces false positives on every TF32+UsePLRPack fixture, Phase 3 cannot pass.

Without resolving 6qib:
- Phase 3 fails on TF32+UsePLRPack fixtures
- Phase 4 (Approach A retirement) is blocked indefinitely
- Design v5 stays at Phase 2

### 4.4 Test surface impact predictions

The "30/90 residuals" column was originally labeled "granularity residuals" — per the probe (§0.7, §2.1), those residuals are actually stream-position cross-iter live-ins exposed by the body-local `latest_writer` walk. Updated:

| Approach | n7og fixtures | 11 reorder/SCC tests | bf16 pin | 30/90 NLL/NGL stream-order residuals |
|---|---|---|---|---|
| Current (identity-tuple) | FAIL (208/624) | PASS | PASS (0) | N/A (masked inside the 208/624) |
| Pure byte-key 3-tuple | PASS (0) | **FAIL** | PASS | N/A (dedup'd out by 3-tuple coarseness) |
| Pure byte-key 6-tuple | FAIL (30/90) | **FAIL** | PASS | EXPOSED as raw extras |
| (a) Compound key + `ordinal_class` | PASS for non-NLL cases, **NLL/NGL residuals reclassify as `OrderInvertedFailure`** | PASS expected | PASS expected | DIAGNOSABLE, not silenced; closure requires (d) |
| (b) Capture alignment (original framing) | PASS likely on register-renaming false positives; NLL residuals **unchanged** because there's no granularity to align | PASS (unchanged) | PASS | NOT addressed |
| (c) Two-phase + filter | PASS expected | PASS expected | PASS expected | N/A (filtered out, masked) |
| (d) Unrolled-program dataflow graph + exemption deletion (probe-derived, expanded) | RESOLVED including NLL/NGL | PASS expected (orthogonal) | PASS | RESOLVED |

The recommended combination is **(a) + (d)** — (a) for the immediate Phase 3 unblock and to surface cross-iter cases as `OrderInvertedFailure`-class findings, then (d) as a separate bead to actually close the body-boundary blind spot. (b) and (c) are both inadequate per the probe.

**Tests currently relying on the exemption (re-examination required under (d)):**

Once the unrolled walk lands and `CMSValidator.py:3831-3843` is deleted, any test whose green run depends on the exemption silently returning `[]` is exposed. Two outcomes are possible per test:

1. **Legitimate cross-iter pattern** — the test exercises the rotating pack-buffer pipelining on `UsePLRPack=True`. Under (d) the unrolled walk resolves the dataflow correctly, no edge is emitted as "missing," the test passes without invoking the exemption. Expected for the n7og BPG#11/oplb fixtures.
2. **Real bug the exemption was masking** — the test passes today only because the exemption silently swallows a finding the validator would otherwise have raised. Under (d) the finding surfaces. Each such case is either (i) a validator bug that needs fixing as part of (d), or (ii) a real CMS schedule defect that's been hidden.

The exemption-classification probe (`/tmp/exemption_probe.py`) gives the baseline: on BPG#11 the exemption fires 192 times and is the only branch that fires for the SHADOW-extras. Other fixtures should be re-probed against the same instrumentation to enumerate the at-risk test surface before deleting the exemption.

### 4.5 Cross-references

- `Tensile/Components/CMSValidator.py:1300` — `DataflowGraph.edge_keys()`
- `Tensile/Components/CMSValidator.py:3626` — current "NOT YET FIXED" comment with bead chain reference
- `Tensile/Components/CMSValidator.py:3735` — `compare_graphs` consumer of edge_keys
- `Tensile/Components/CMSValidator.py:3831-3843` — **cross-subiter ALU-producer exemption**; anti-pattern per §0.8; flagged for DELETION as part of (d) once the unrolled-program dataflow graph lands. Empirical: absorbs 100% (192/192) of BPG#11's SHADOW-extras today. Do NOT extend this code or thread additional carve-outs through it — every new condition piles more surface-pattern matching on a foundation §0.8 already identifies as broken.
- `Tensile/Components/ScheduleCapture.py:1428` — `_byte_keys_for_resource`
- `Tensile/Components/ScheduleCapture.py:1481` — `_resolve_producers` (the byte-grouping function originally suspected as the residual cause; ruled out by `n7og_PROBE_REPORT.md`)
- `Tensile/Tests/unit/test_n7og_edge_keys_multifixture.py` — the empirical probe (3 fixtures, `xfail strict` on 2)
- `Tensile/Components/n7og_PROBE_PLAN.md` — falsification plan for the per-iter-duplication hypothesis
- `Tensile/Components/n7og_PROBE_REPORT.md` — empirical results; identifies the body-local `latest_writer` walk as the actual residual mechanism (§0.7 here)
- `Tensile/Components/DEFAULT_SCHEDULER_REFERENCE_DESIGN.md` §3 — comparison contract (says SHADOW and CMS should differ ONLY in scheduling)
- `rocm-libraries-r62g` — Phase 3 hard gate (blocked by 6qib)
- `rocm-libraries-udqg` — superseded mechanism description
- `rocm-libraries-32tg` — superseded refinement (3-tuple complete fix claim)

## 5. Recommendation

**Approach (a) — Compound key with `ordinal_class` — PLUS a separate, larger P0 bead for (d): rewrite `build_dataflow_graph` to walk the unrolled program AND delete the cross-subiter ALU-producer exemption at `CMSValidator.py:3831-3843`.**

Justification (updated per `n7og_PROBE_REPORT.md` and the exemption-classification probe):
- (a) directly addresses both requirements (allocation-invariant + order-sensitive)
- Smallest implementation surface for the immediate Phase 3 unblock
- Doesn't introduce the centralized-list anti-pattern (Approach c)
- Doesn't take on unbounded capture-layer scope (Approach b — moot anyway per the probe)
- Reframes the 30/90 NLL/NGL residuals from "silent false-positive inflation" into "`OrderInvertedFailure`-classified findings" — diagnosable, not silenced
- The body-local-walk blind spot (the actual underlying cause per the probe) becomes a separate, scoped P0 bead — not absorbed into 6qib
- (d) is now substantially broader than the original "extend Phase 2 `latest_writer` across body boundaries" framing because the probe shows the cross-subiter ALU-producer exemption (§0.8) is a load-bearing workaround that exists *only* to mask the same blind spot. The principled close-out is to model the dataflow correctly and remove the workaround, not patch around it.

The decision sub-question for Approach (a) is what `ordinal_class` should be. Options:
- (a.1) Binary: producer-before-consumer / consumer-before-producer
- (a.2) Pair: `(prod_emission_ordinal, cons_emission_ordinal)` — full information but reintroduces some allocation-coupling
- (a.3) Stride: position of consumer relative to producer in subiter scope — middle ground

(a.1) is the simplest and is sufficient for `OrderInvertedFailure` detection (which is binary). (a.2) is the safest for catching subtle reorder cycles but breaks the simplicity. (a.3) is a hedge.

Suggest starting with (a.1) and only escalating if Phase 3 surfaces a case (a.1) misses.

### 5.1 The (d) bead — build the dataflow graph over the unrolled program

**Framing.** If you mentally unroll the loop into one long linear instruction stream, every read maps to its most-recent prior write. Body labels (`PRO`, `ML`, `NGL`, `NLL`) and subiter indices are code-generation organization, not dataflow boundaries. The existing body-local `latest_writer` walk (§0.7) artificially partitions that stream at body boundaries and so cannot resolve cross-body / cross-iter live-ins. The cross-subiter ALU-producer exemption (§0.8) exists *because* the walk can't reach across body boundaries to find the legitimate cross-iter producer; deleting the exemption alone would simply re-route the same 192 BPG#11 extras into `OrderInvertedFailure` findings rather than `[]`. The fix is to make the model reflect the program.

**Implementation.**

1. **Concatenate the body streams in execution order**, producing one logical unrolled stream:
   - `PRO` (prologue)
   - For each `ML` iter (0 .. `LoopIters - 1`): the body's instructions in their stream order
   - `NGL` (no-global-load tail)
   - For each `NLL` iter (0 .. `NoLoadLoopIter - 1`): the body's instructions in their stream order
   - `POST` (epilogue)
   The unroll factor is `LoopIters` for `ML` and `NoLoadLoopIter` for `NLL`, taken from the same kernel parameters that drive code generation. Each per-iter copy of the body contributes the SAME captured node objects (no GraphNode duplication); only the stream-position index is rewritten to reflect the unrolled position.
2. **Single `latest_writer` walk over the unrolled stream.** Initialize empty at the start of the unrolled program; never reset at a body or iter boundary. Process every write and read in unrolled-stream order. Every read resolves to its most-recent prior write across the unrolled program. Body labels and subiter indices are preserved on the resulting edges purely as diagnostic annotations.
3. **Body labels become diagnostic annotations only.** Edge `body_label` (producer / consumer) is still useful for `cms_node_label` and `OrderInvertedFailure.iter_delta` reporting, but it no longer gates dataflow resolution. `iter_delta` is computed from the unrolled-stream indices, not from body-membership.
4. **The cross-subiter ALU-producer exemption at `CMSValidator.py:3831-3843` becomes dead code and is DELETED as part of (d).** The exemption is not preserved alongside the unrolled walk; it is removed because the invariant it was protecting (cross-iter rotating-pack-buffer dataflow is legitimate) is now captured by the model itself — the legitimate producer is the most-recent prior writer in the unrolled stream, the edge resolves cleanly, `compare_graphs` sees identical edge sets, and no exemption needs to fire. Deleting the exemption and leaving the body-local walk in place would be strictly worse than the status quo (192 extras become 192 `OrderInvertedFailure` findings); both pieces must land together.
5. **Phase 1 `OrderInvertedFailure` detection still works.** A real reorder produces a SHADOW edge whose producer-then-consumer positions invert in the CMS unrolled stream — the byte-key + ordinal_class comparison (under (a)) catches it the same way it would today, because the actual dataflow producer is still being resolved correctly on both sides.

**Why this is the right framing rather than "extend latest_writer across body boundaries":** the body-local walk's blind spot is one symptom; the cross-subiter exemption is another. Both arise from the same representational mistake of treating codegen organization (bodies, subiters) as dataflow structure. A patch that only seeds `latest_writer` at body boundaries leaves the exemption alive and the next codegen change that creates a new cross-body pattern will need its own carve-out. The unrolled-program walk eliminates the class of problem.

**Scope estimate.** ~200-400 LOC in `build_dataflow_graph` and the supporting `SchedulePosition` / `latest_writer` machinery, plus deletion of the 13-line exemption block at `CMSValidator.py:3831-3843`. Plus test fixture updates: each test currently relying on the exemption silently returning `[]` needs re-examination (per §4.4). The exemption-classification probe at `/tmp/exemption_probe.py` is the baseline tool — re-run against each fixture to enumerate which tests are at risk before deleting the exemption.

**Acceptance criterion.** The 192 extra-in-SHADOW edges in BPG#11's NLL and the 16 missing-in-SHADOW edges in NGL all resolve to 0. The bf16 negative pin (`UsePLRPack=False`) continues to pass with 0 mismatches. The 11 reorder/SCC/carveout tests pass (their detection paths fire as designed under the unrolled walk + (a)'s compound key). No test that previously passed via exemption-silencing transitions to a failure unless that failure is investigated and either fixed or recognized as a legitimate finding.

Filed P0 with `br dep add r62g <new-bead>`.
