# Implementation Plan — C3d (rocm-libraries-xxj4): `edge_keys()` byte-key basis migration

**Status:** planning complete — ready for implementation  
**Bead:** `rocm-libraries-xxj4`  
**Depends on:** `rocm-libraries-1rsy` (C3c, closed)  
**Blocks:** `rocm-libraries-67us` (C3e), `rocm-libraries-si5f` (C3h)  
**Working tree:** `/home/alvasile/rocm-libraries/.worktrees/validator_long_term_plans/`

---

## §1 Scope

C3d migrates `DataflowGraph.edge_keys()` from the identity-tuple basis `(producer.identity, consumer.identity, edge_kind, intra_operand_byte_offset, src_operand_slot, sink_operand_slot)` to the byte-key basis `(producer_write_byte_keys, consumer_read_byte_keys, edge_kind, intra_operand_byte_offset, src_operand_slot, sink_operand_slot)`. The byte-key fields are already populated on every `DataflowEdge` at edge-formation time (C3c landed `producer_write_byte_key` and `consumer_read_byte_key`); C3d simply substitutes them into the keying tuple. `compare_graphs` builds its `ref_edges_by_key` index using the same shape as `edge_keys()`, so that index construction also changes. `diagnose_missing_edge` uses identity to look up `p_node`/`c_node` in the subject graph; that lookup **stays identity-based** in C3d because `diagnose_missing_edge`'s full Phase 0/1 migration is C3f's scope. One caller (`test_approach_a_non_cms_reference.py`) currently pins the exact Phase-0 error message that fires when identity lookup fails for a missing edge — C3d must re-examine whether that test still fires (it will, but via a different path), and re-pin the assertion accordingly. The n7og xfail markers stay through C3d; removal is C3h.

---

## §2 Investigation Findings

### A — Current `edge_keys()` shape

`DataflowGraph.edge_keys()` at `CMSValidator.py:1248-1322` returns:

```python
{(e.producer.identity, e.consumer.identity,
  e.edge_kind, e.intra_operand_byte_offset,
  e.src_operand_slot, e.sink_operand_slot)
 for e in self.edges}
```

`producer.identity` and `consumer.identity` are each a 3-tuple `(canonical_render, source_module_id, emission_ordinal)` (Approach 2 / rocm-libraries-dfd8). The `canonical_render` embeds the literal rendered register operand names, making the key allocation-sensitive. When SHADOW and CMS spell the same physical register differently (`vgprValuA_T0_I0+0` vs `vgprValuA_X0_I0+12`) the identity differs and edge_keys diverge falsely.

Identity references in the keying basis: `e.producer.identity` and `e.consumer.identity`. Both disappear in C3d.

### B — The new keying basis: field-by-field analysis

New key shape:
```python
(producer_write_byte_keys, consumer_read_byte_keys, edge_kind,
 intra_operand_byte_offset, src_operand_slot, sink_operand_slot)
```

**`producer_write_byte_keys`** — the byte-key tuple the edge's producer wrote, stored as `DataflowEdge.producer_write_byte_key` (note: singular field name, plural semantics — it's already a tuple of byte-key pairs). Populated at edge-formation time (`CMSValidator.py:2200`) by calling `_byte_keys_for_resource(overlap, name_to_idx=p_n2i)`. Allocation-invariant: `vgprValuA_T0_I0+0` and `vgprValuA_X0_I0+12` both resolve to `('v', 12)` when `name_to_idx` is populated (the whole point of C3c's `name_to_idx` threading). Type: `tuple` of `(regType, regIdx)` or `(regType, name, offset)` pairs. Already hashable.

**`consumer_read_byte_keys`** — the byte-key tuple the edge's consumer read, stored as `DataflowEdge.consumer_read_byte_key`. Same shape. Populated at edge-formation time (`CMSValidator.py:2201`) by calling `_byte_keys_for_resource(overlap, name_to_idx=n2i)`.

**`edge_kind`** — string: `'raw_intrawave'`, `'lds_raw_intrawave'`, `'lr_to_gr_lds_reuse'`, `'gr_to_lr_lds_reuse'`. Already on `DataflowEdge.edge_kind`. Unchanged.

**`intra_operand_byte_offset`** — allocation-invariant tuple of byte positions WITHIN the consumer's read operand (e.g. `(0,)`, `(0, 1, 2, 3)`). Already on `DataflowEdge.intra_operand_byte_offset`. Unchanged.

**`src_operand_slot`** — integer positional index of the producer's write operand. Already on `DataflowEdge.src_operand_slot`. Unchanged.

**`sink_operand_slot`** — integer positional index of the consumer's read operand. Already on `DataflowEdge.sink_operand_slot`. Unchanged.

**Hashability:** `DataflowEdge.producer_write_byte_key` and `consumer_read_byte_key` are `tuple` objects (constructed by `_byte_keys_for_resource` which always returns `tuple`). Tuples are hashable as long as their elements are hashable. Each element is either `(str, int)`, `(str, str, int)`, or `("mem", str, id, int)` — all hashable. The full 6-tuple is therefore hashable and set-usable with no changes.

### C — How edges get byte_key data

`build_dataflow_graph` Phase 2 in `CMSValidator.py:2180-2210` already populates both `producer_write_byte_key` and `consumer_read_byte_key` on every `DataflowEdge` at edge-formation time. The fields were added in C3c (as "diagnostic annotation fields, not in edge_keys()"). C3d promotes them into the keying tuple — zero new data computation required. For barrier edges (`lr_to_gr_lds_reuse` / `gr_to_lr_lds_reuse`) the byte-key fields are also set at `CMSValidator.py:2441-2442`.

`_resolve_producers` (`ScheduleCapture.py:1643`) does NOT carry byte-keys through its yield; it yields `(writer_node, overlap, intra_offsets, write_slot)`. The byte-key computation happens at the call site in `build_dataflow_graph` using `_byte_keys_for_resource(overlap, ...)` — this is already done in C3c.

### D — Iter-blind keys: pipelining cancellation walk-through

BPG#11 (`UsePLRPack=True`, `UseCustomMainLoopSchedule=1`):

- Unrolled stream (SHADOW): `... ML_iter[1] writes via PackB3-pipelined (byte_key ('v',31..34)) ... NLL MFMA reads ('v',31..34) ...`
  - Producer byte_key: `(('v',31), ('v',32), ('v',33), ('v',34))`
  - Consumer byte_key: same
  - `edge_kind = 'raw_intrawave'`, `intra = (0,1,2,3)`, `src_slot=0`, `sink_slot=0`
  - edge_key = `((('v',31),...), (('v',31),...), 'raw_intrawave', (0,1,2,3), 0, 0)`

- Unrolled stream (CMS): SAME physical instructions, same byte-keys. `latest_writer` now populated by ML_iter[1]'s PackB3 writes BEFORE NLL MFMA reads because the unrolled walk goes ML_iter[0] → ML_iter[1] → NGL → NLL in sequence.
  - Producer byte_key: identical
  - Consumer byte_key: identical
  - Same `edge_kind`, `intra`, slots
  - edge_key: byte-equal to SHADOW's

Result: key is identical on both sides → set-diff cancels → 0 mismatches. This is why C3d (not just C3c) is needed: C3c fixed the `latest_writer` walk; C3d fixes the key shape so the resulting identical edges actually cancel.

### E — `edge_kind` source

`edge_kind` is a field on `DataflowEdge` — a `str` with value `'raw_intrawave'`, `'lds_raw_intrawave'`, `'lr_to_gr_lds_reuse'`, or `'gr_to_lr_lds_reuse'`. Assigned at edge-construction time inside `build_dataflow_graph`. Already in the existing 6-tuple; no change in C3d.

There is no external enum or classification enum — it is a plain string. No import needed.

### F — Hashability

Both byte-key fields are `tuple` objects from `_byte_keys_for_resource`. A `tuple` of `(str, int)` or `(str, str, int)` pairs is natively hashable. The 6-tuple `(tuple_of_tuples, tuple_of_tuples, str, tuple_of_ints, int, int)` is hashable. No `frozenset` needed.

Consistent choice: keep `tuple` (same as the existing fields). The existing `intra_operand_byte_offset` is already a `tuple`, so the key shape is uniform.

### G — Identity references to remove

After C3d, `edge_keys()` body contains ZERO identity references. The two call sites:
1. `CMSValidator.py:1319-1321` — `edge_keys()` return statement: `e.producer.identity` and `e.consumer.identity` replaced by `e.producer_write_byte_key` and `e.consumer_read_byte_key`.
2. `CMSValidator.py:3798-3800` — `ref_edges_by_key` construction in `compare_graphs`: same substitution.

Residual identity uses (NOT in C3d scope — they stay):
- `CMSValidator.py:3832-3835` — `diagnose_missing_edge` Phase 0 lookup by identity (C3f scope)
- `CMSValidator.py:3872-3875` — defensive identity fallback in `diagnose_missing_edge` (C3f scope)
- `CMSValidator.py:3933-3935` — SCC clobber search by consumer identity (C3f scope)

### H — n7og probe current behavior [CORRECTED]

`test_n7og_edge_keys_multifixture.py` has two xfail (`strict=True`) fixtures:
- `bpg11-tf32-4x4-tn`: 208 mismatches under current identity-tuple basis.
- `oplb-tf32-6x8-tn`: 624 mismatches.

**These xfails will NOT flip to XPASS after C3d.** The mismatches are caused by the SHADOW capture pipeline's broken `name_to_idx` (missing bindings for rotating T/X pack-buffer registers under `UsePLRPack+UseMFMAF32XEmulation`), which causes `_byte_keys_for_resource` to return `(-1,)` sentinels. The source itself states this explicitly at `CMSValidator.py:3718-3722`: "switching `edge_keys` to byte-keys leaves the mismatch in place because the byte-keys themselves are wrong on the SHADOW side." C3d changes the matching basis; it does NOT fix the underlying broken SHADOW byte-keys. Those broken byte-keys will still produce mismatches (now directly visible in the byte-key tuples rather than via identity-tuple divergence). The xfails remain as XFAIL (still failing, still correctly marked) after C3d. No change to these markers in C3d.

The xfails are tied to `rocm-libraries-udqg` (the SHADOW pipeline fix), NOT to C3d. Removal is deferred to C3h (`rocm-libraries-si5f`) per bead acceptance — but that removal is gated on udqg landing, not on C3d landing.

`bf16-256x256x64-tn`: 0 mismatches currently (positive pin). Stays 0 after C3d.

### I — xfail handling in C3d [CORRECTED]

The n7og test xfails are cited against `rocm-libraries-udqg` and will remain XFAIL (still failing, still correctly marked) after C3d. **Zero changes to `test_n7og_edge_keys_multifixture.py` xfail markers in C3d.** The `strict=True` is correct: it ensures XPASS surfaces loudly if udqg lands before C3h cleans up the markers. Do not flip to `strict=False` — that would suppress the XPASS signal and violate the bead's "surfaces honestly" contract. Leave the markers exactly as-is.

### J — bf16 negative pin

The bf16 `256x256x64-tn` fixture in `test_n7og_edge_keys_multifixture.py` has no xfail marker — it asserts 0 mismatches directly. It currently passes. After C3d it must continue to pass. The byte-key for this fixture's edges will be numerically resolved (no `UsePLRPack` rotating buffer ambiguity), and the unrolled walk (already landed in C3c) will produce identical edge sets on SHADOW and CMS. No action needed beyond confirming in the verification step.

### K — Backward-compatibility of consumers [CORRECTED — caller inventory expanded]

Complete enumeration of `edge_keys()` callers (verified by grep):

| Location | Usage | C3d action |
|---|---|---|
| `CMSValidator.py:3772-3773` | `compare_graphs` calls `reference.edge_keys()` and `subject.edge_keys()` → set-diff | No change to call; key shape changes |
| `CMSValidator.py:3798-3800` | `ref_edges_by_key` construction in `compare_graphs` | Must update key construction to match new `edge_keys()` shape |
| `test_n7og_edge_keys_multifixture.py:309` | Direct `graph.edge_keys()` call in the probe | No structural change; mismatch count unchanged (xfails stay XFAIL — see §H/§I) |
| `test_approach_a_non_cms_reference.py:287-307` | Indirect via `compare_graphs`; test pins Phase-0 CaptureConsistencyError | Re-examine and re-pin (see §5 below) |
| `test_dataflow_graph_comparison.py:1017-1018,1044` | Direct `g.edge_keys()` calls — asserts equality on identical-allocation, inequality on distinct-allocation | Assertions remain correct under byte-key basis (identical allocation → identical byte-keys; distinct allocation → distinct byte-keys). Comment on line 3144 is stale ("different identities → different edge keys" should read "different byte-keys → different edge keys"). Update the comment; assertion logic unchanged. |
| `test_dataflow_graph_register_gaps.py:3146` | Direct `g.edge_keys()` call — asserts inequality for distinct-allocation VSwap pair | Same as above: assertion correct, in-line comment stale. Update comment. |
| `repro_cross_subiter_artifact.py:450-451` | Diagnostic script only; uses `edge_keys()` for edge-set diff display | No action needed; opaque key use, no assertion on key shape. |

No callers destructure the tuple by position (e.g., `key[0]`, `key[1]`). All callers treat the tuple as an opaque key for set membership. The shape change is invisible to callers except for:
1. The `ref_edges_by_key` construction site in `compare_graphs` (which mirrors the `edge_keys()` tuple) — must update.
2. Two test-file comments that describe the old "identity" discriminator — must update to "byte-key" wording (no assertion logic changes).

**`test_approach_a_non_cms_reference.py` re-examination:**

This test currently pins that `compare_graphs` raises `CaptureConsistencyError` with message `"identity-coverage check at compare_graphs entry was bypassed"`. This message is emitted by `diagnose_missing_edge` Phase 0 when `p_node = next((n for n in reversed(subj_graph.nodes) if n.identity == p_id), None)` returns `None`.

Under C3d: `compare_graphs` builds `missing_keys = ref_keys - subj_keys` using the NEW byte-key basis. If the byte-keys on both sides are consistent (both see the same physical bytes), the missing_keys set will be empty and `diagnose_missing_edge` is never called. If the byte-keys still diverge (due to remaining sentinel keys from unresolved `name_to_idx`), there will be missing keys but the `p_id` / `c_id` lookup will still run in `diagnose_missing_edge` (because Phase 0 hasn't changed yet). The test for the `non_cms_reference` fixture uses the BPG#11 config (`UsePLRPack=True`) against a non-CMS reference build — whether the byte-keys match depends on whether the non-CMS reference's `name_to_idx` resolves the T/X registers correctly.

**The `test_approach_a_non_cms_reference.py` test may flip behavior in C3d.** Two scenarios:
- If the non-CMS reference build has correct `name_to_idx` (name_to_idx populated by C3c's threading), byte-keys match, no missing edges, no `CaptureConsistencyError` → the test's `pytest.raises(CaptureConsistencyError)` block would FAIL.
- If the non-CMS reference build's captures still produce sentinel byte-keys, missing edges remain, `diagnose_missing_edge` fires, Phase 0 raises `CaptureConsistencyError` → test continues to pass.

This needs empirical verification during implementation. If the test flips, the new correct assertion is "no failures" or a different error class. File a new bead if this is a real behavioral change that needs classification.

---

## §3 Design — new key tuple shape

```python
# New edge_keys() return shape
(
    e.producer_write_byte_key,    # tuple of byte-key pairs, e.g. (('v',12), ('v',13))
    e.consumer_read_byte_key,     # tuple of byte-key pairs, e.g. (('v',12), ('v',13))
    e.edge_kind,                  # str: 'raw_intrawave' | 'lds_raw_intrawave' | ...
    e.intra_operand_byte_offset,  # tuple of ints: byte positions within the read operand
    e.src_operand_slot,           # int: producer's positional write slot
    e.sink_operand_slot,          # int: consumer's positional read slot
)
```

**Allocation-invariance:** `producer_write_byte_key` and `consumer_read_byte_key` are resolved at edge-formation time by `_byte_keys_for_resource(overlap, name_to_idx=...)`. The `name_to_idx` resolver collapses symbolic-vs-numeric naming (C3c already threads this). Numeric byte-keys (`('v', 15)`) are identical regardless of symbolic name used by the instruction.

**Iter-blindness:** byte-key tuples are derived from physical register indices, not from which iter copy the instruction belongs to. ML iter 0 and ML iter 1 of the same PackB3 instruction produce the same `producer_write_byte_key` → same edge_key → cancels in set-diff.

**Hashability:** `tuple` of `tuple` of primitive values is natively hashable. No conversion needed.

**Type consistency:** field name `producer_write_byte_key` (already on `DataflowEdge`) is singular but is typed as `tuple` containing potentially multiple byte-keys. The keying uses the field directly — no rename needed.

---

## §4 `edge_keys()` rewrite

The rewrite is mechanical — two tuples replace two identity tuples:

**Before (current):**
```python
def edge_keys(self):
    return {(e.producer.identity, e.consumer.identity,
             e.edge_kind, e.intra_operand_byte_offset,
             e.src_operand_slot, e.sink_operand_slot)
            for e in self.edges}
```

**After (C3d):**
```python
def edge_keys(self):
    return {(e.producer_write_byte_key, e.consumer_read_byte_key,
             e.edge_kind, e.intra_operand_byte_offset,
             e.src_operand_slot, e.sink_operand_slot)
            for e in self.edges}
```

The docstring must be updated to reflect the new tuple shape. Key points to document:
- `producer_write_byte_key` / `consumer_read_byte_key`: allocation-invariant physical byte-key tuples, already on every `DataflowEdge`, resolved at edge-formation time.
- No identity-tuple references remain in the keying basis.
- Iter-blind: ML iter copies of the same instruction produce identical byte-keys → set-diff cancels cross-iter pipelined edges as desired.
- The existing `producer.identity` / `consumer.identity` fields on the nodes remain available as diagnostic annotations but are NOT in the keying tuple.

---

## §5 Caller migration

### 5.1 `compare_graphs` — `ref_edges_by_key` construction

`CMSValidator.py:3796-3801`. Currently:
```python
ref_edges_by_key = {}
for e in reference.edges:
    key = (e.producer.identity, e.consumer.identity,
           e.edge_kind, e.intra_operand_byte_offset,
           e.src_operand_slot, e.sink_operand_slot)
    ref_edges_by_key.setdefault(key, e)
```

After C3d:
```python
ref_edges_by_key = {}
for e in reference.edges:
    key = (e.producer_write_byte_key, e.consumer_read_byte_key,
           e.edge_kind, e.intra_operand_byte_offset,
           e.src_operand_slot, e.sink_operand_slot)
    ref_edges_by_key.setdefault(key, e)
```

The comment block at `CMSValidator.py:3776-3794` must be updated to reflect byte-key basis.

### 5.2 `diagnose_missing_edge` — Phase 0 identity lookup: NO CHANGE in C3d

`CMSValidator.py:3832-3843`. The Phase 0 lookup (`p_node = next(...)`, `c_node = next(...)`) stays identity-based in C3d. The full Phase 0/1 migration is C3f. This means that when a "missing key" (under the new byte-key basis) is found in `compare_graphs`, `diagnose_missing_edge` will look up the ref_edge's producer/consumer by their identity tuples in `subj_graph.nodes`. This continues to work because:
- The subject graph's nodes still carry `identity` fields.
- The identity lookup in Phase 0 is separate from the edge-keying basis.

**Consequence for `test_approach_a_non_cms_reference.py`:** Needs empirical verification during implementation. If the byte-key migration causes the T/X register-naming case to produce empty missing_keys (because byte-keys now match), the test's `pytest.raises(CaptureConsistencyError)` block will fail. The implementer must run the test and re-pin accordingly. See §9 Risk 1.

### 5.3 Defensive identity fallback in `diagnose_missing_edge`: NO CHANGE in C3d

`CMSValidator.py:3864-3878`. The block that scans `subj_graph.edges` matching by `e.producer.identity == p_id and e.consumer.identity == c_id and ...` stays as-is in C3d. This is C3f scope.

### 5.4 `test_n7og_edge_keys_multifixture.py` xfail updates [CORRECTED — no changes]

The two TF32 xfail entries will NOT XPASS under C3d. The mismatches originate in the SHADOW pipeline's broken `name_to_idx` (the `rocm-libraries-udqg` root cause), not in the edge-key basis. C3d changes the keying basis from identity to byte-key, but the underlying byte-keys on the SHADOW side are still broken (`(-1,)` sentinels), so the byte-key diff will still diverge. The xfails remain as XFAIL after C3d.

**No changes to `test_n7og_edge_keys_multifixture.py` in C3d.** The `strict=True` markers are correct and must remain. Removal is C3h scope, gated on udqg landing.

---

## §6 Test impact [CORRECTED]

### Tests that STAY XFAIL (unchanged by C3d)

- `test_n7og_edge_keys_multifixture.py::test_shadow_vs_cms_edge_keys_match[bpg11-tf32-4x4-tn]` — remains XFAIL (208 mismatches still present; root cause is SHADOW pipeline's broken `name_to_idx`, not the key basis). No marker changes.
- `test_n7og_edge_keys_multifixture.py::test_shadow_vs_cms_edge_keys_match[oplb-tf32-6x8-tn]` — same (624 mismatches persist).

### Tests that STAY PASSING

- `test_n7og_edge_keys_multifixture.py::test_shadow_vs_cms_edge_keys_match[bf16-256x256x64-tn]` — 0 mismatches before and after
- `test_dataflow_graph_comparison.py::TestEdgeIdentityByteKeyContract` — both tests (identical-allocation equality; distinct-allocation inequality) continue to hold. The commented rationale mentions "different identities" — update to "different byte-keys" but no assertion change.
- `test_dataflow_graph_register_gaps.py` VSwap inequality test — assertion holds; comment update needed.
- All reorder/SCC/carveout tests that test `OrderInvertedFailure` and similar: these fire from `diagnose_missing_edge` which still uses identity lookup (unchanged in C3d). Their behavior is unaffected.

### Tests that NEED RE-EXAMINATION

- `test_approach_a_non_cms_reference.py::test_non_cms_reference_compare_graphs_surfaces_only_known_residuals` — pins `CaptureConsistencyError` with "identity-coverage check at compare_graphs entry was bypassed". After byte-key migration, if the byte-keys on both sides diverge (as they do for this fixture — non-CMS reference vs CMS), `compare_graphs` will still call `diagnose_missing_edge` which still uses identity lookup in Phase 0. Whether the Phase-0 raise fires depends on whether the missing-key producer/consumer identities exist in the subject graph. Empirically verify. Re-pin or file bead as appropriate (see §9 Risk 1).

### Tests with comment-only staleness (no assertion change)

- `test_dataflow_graph_comparison.py:3144`: comment says "different identities → different edge keys" — stale after C3d; update to "different physical bytes → different byte-keys → different edge keys".
- `test_dataflow_graph_register_gaps.py:3144-3146`: same comment staleness.

### xfail markers

- `test_n7og_edge_keys_multifixture.py` TF32 fixtures: **NO CHANGES.** `strict=True` stays. Markers stay. Removal is C3h scope.

---

## §7 Step-by-step implementation order

1. **Update `DataflowGraph.edge_keys()` docstring** to describe the new byte-key basis and why identity is no longer in the key. Remove references to "why `producer.identity` instead of byte-key."

2. **Rewrite `edge_keys()` return statement** (one-liner change at `CMSValidator.py:1319-1321`): replace `e.producer.identity, e.consumer.identity` with `e.producer_write_byte_key, e.consumer_read_byte_key`.

3. **Update `compare_graphs` `ref_edges_by_key` construction** (`CMSValidator.py:3796-3801`): same two-field substitution. Update the surrounding comment block to reflect byte-key basis.

4. **Update stale comments in `test_dataflow_graph_comparison.py` and `test_dataflow_graph_register_gaps.py`**: find the comment that says "different physical registers in canonical_render -> different producer/consumer identities -> different edge keys" and update to "different physical registers -> different byte-keys -> different edge keys". No assertion changes.

5. **Run `pytest Tensile/Tests/unit/test_n7og_edge_keys_multifixture.py`** — confirm: two TF32 fixtures remain XFAIL (no change), bf16 fixture passes. The TF32 xfails will NOT flip to XPASS (byte-key migration does not fix the SHADOW pipeline's broken name_to_idx).

6. **Run `pytest Tensile/Tests/unit/test_approach_a_non_cms_reference.py`** — confirm the test behavior. If it flips (the `CaptureConsistencyError` no longer fires because byte-keys now cancel the missing edges), re-pin the assertion to match the new behavior. If the test now asserts `failures == []`, that is correct and represents a real improvement. If it raises a different exception, investigate.

7. **Run the full unit suite** (`pytest Tensile/Tests/unit/ --ignore=...test_MatrixInstructionConversion.py`). Classify every new failure:
   - Unexpected regression: investigate immediately; file bead if a real bug surfaces.
   - Test pinning old identity-based key shape in assertions: re-pin to byte-key shape.
   - Test exposing a new classification gap: file bead.

8. **Verify acceptance criteria** (see §8).

---

## §8 Validation

### n7og probe expectations [CORRECTED]

| Fixture | Before C3d | After C3d |
|---|---|---|
| `bpg11-tf32-4x4-tn` | XFAIL (208 mismatches) with `strict=True` | XFAIL (mismatches persist — SHADOW byte-keys still broken, udqg not landed). No marker change. |
| `oplb-tf32-6x8-tn` | XFAIL (624 mismatches) with `strict=True` | XFAIL (mismatches persist — same root cause). No marker change. |
| `bf16-256x256x64-tn` | PASS (0 mismatches) | PASS (0 mismatches) unchanged |

### bf16 negative pin expectations

`test_n7og_edge_keys_multifixture.py::test_shadow_vs_cms_edge_keys_match[bf16-256x256x64-tn]` asserts 0 mismatches. Continues to pass after C3d.

### Overall failure delta vs end-of-C3c (22 FAILED + 2 ERROR baseline) [CORRECTED]

C3d does NOT resolve the two TF32 xfail mismatches (those require udqg). The failure count from C3d alone: no improvement to the 22+2 baseline from the n7og fixtures. The net change comes only from any test_approach_a_non_cms_reference.py behavioral shift (empirical verification required in step 6).

**The bead's acceptance criteria [CORRECTED]:**
- `edge_keys()` is byte-key based: confirmed — `e.producer_write_byte_key` and `e.consumer_read_byte_key` are the first two elements.
- No identity-tuple references in the keying basis: confirmed — the `edge_keys()` one-liner and the `ref_edges_by_key` construction both use byte-key fields only.
- n7og probe BPG#11 + oplb fixtures: ref − subj direction resolves to 0 mismatches under new keying: **NOT confirmed by C3d alone.** The source explicitly states (CMSValidator.py:3718-3722) that byte-key migration does NOT resolve these mismatches because the SHADOW pipeline's byte-keys are broken (`(-1,)` sentinels). The xfails will remain XFAIL after C3d. This acceptance criterion in the bead is unachievable by C3d alone — it requires udqg to also land. The bead's acceptance wording appears to have been written with incorrect assumptions about what C3d would fix. The implementer should note this discrepancy in the commit message and not consider C3d "complete" for the n7og acceptance criterion until udqg lands. C3d is still the right change for the principled basis migration — the criterion gap is in the bead description, not in the implementation scope.
- bf16 negative pin: 0 mismatches: confirmed unchanged.

---

## §9 Risks and Open Questions

### Risk 1 — `test_approach_a_non_cms_reference.py` behavior flip (MEDIUM)

This test pins `CaptureConsistencyError` with `"identity-coverage check at compare_graphs entry was bypassed"`. The message fires from `diagnose_missing_edge` Phase 0 when a missing key's producer/consumer identity is not found in `subj_graph.nodes`.

Under byte-key basis: if the non-CMS reference's edges and the CMS subject's edges have matching byte-keys (because C3c's `name_to_idx` threading resolves the T/X registers), there will be NO missing keys, `diagnose_missing_edge` is never called, and the `CaptureConsistencyError` is never raised. The test's `pytest.raises(CaptureConsistencyError)` block would then fail.

**This may represent the test correctly becoming green** — the byte-key migration actually fixes the underlying divergence the test was documenting. If so, the correct re-pin is to assert `failures == []`.

**Resolution:** Run the test empirically during implementation. Re-pin to actual behavior. If the test now passes cleanly, that is an improvement, not a regression — document it in the commit message.

### Risk 2 — Sentinel byte-keys in barrier edges (LOW)

`lr_to_gr_lds_reuse` and `gr_to_lr_lds_reuse` edges set `producer_write_byte_key` via `_byte_keys_for_resource(resource)` without `name_to_idx` (`CMSValidator.py:2405`, `2441`). If `resource` is a `MemoryRegion`, byte-keys are `("mem", space, buffer_id, offset)` — numeric, allocation-invariant. If `resource` is a symbolic `RegisterContainer` without `name_to_idx`, the key falls through to the symbolic form `(rt, name, offset)`. For barrier edges the resources are LDS memory regions, so this path is not hit. Low risk, but worth confirming edge_key of a barrier edge looks correct in a debug run.

### Risk 3 — Empty byte-key tuples (LOW)

`_byte_keys_for_resource` returns `()` for unrecognized resource shapes. An edge with `producer_write_byte_key = ()` produces a key where the first element is an empty tuple. This is hashable and valid but may cause unexpected set-diff behavior (two edges with empty byte-keys on both sides share the same key, which may cause merging). Under C3b's `byte_key_writers`, every write must have at least one byte-key or it wouldn't be tracked. The risk is theoretical for the current instruction set.

### Risk 4 — docstring stale references (LOW)

The `edge_keys()` docstring is extensive (~60 lines). It contains multiple references to the identity-tuple rationale ("Why `producer.identity` instead of the memo's literal byte-key proposal"). After C3d these explanations are inverted. The full docstring must be rewritten to explain the byte-key basis.

### Risk 5 — compare_graphs docstring (LOW)

`CMSValidator.py:3644` currently says `"(producer.identity, consumer.identity, register, edge_kind)"`. Must update to `"(producer_write_byte_key, consumer_read_byte_key, edge_kind, ...)"`.

### Risk 6 — Bead acceptance criterion unachievable by C3d alone (HIGH, KNOWN)

The bead `rocm-libraries-xxj4` acceptance says "n7og probe BPG#11 + oplb fixtures: ref − subj direction resolves to 0 mismatches under new keying." This is not achievable by C3d alone — the SHADOW pipeline's broken `name_to_idx` (udqg scope) is the actual blocker. The byte-key migration (C3d) is still the correct principled change; it does not introduce a regression. The discrepancy should be noted in the commit message so the bead audit trail is clear. The bead can be partially accepted (key basis migration criteria met; n7og probe criteria deferred to udqg+C3h).

---

## §10 New beads to file

### Potential bead: `test_approach_a_non_cms_reference.py` re-pin scope

If the `test_approach_a_non_cms_reference.py` test behavior changes in C3d (either passing cleanly or raising a different error), the new behavior must be examined:
- If it passes cleanly (0 failures): this is net improvement. Re-pin, note in commit message. No new bead needed.
- If it raises a different error class (not `CaptureConsistencyError`): this is a new classification gap. File a bead with `br dep add` to block C3h. Investigate whether it indicates a real dataflow inconsistency or a validator gap.
- If it raises the SAME `CaptureConsistencyError` but with different message text: re-pin the assertion to the new message.

The implementer must make this determination empirically and either close inline or file a bead per the no-deferred-discoveries rule.

No other new beads are anticipated from C3d's scope. C3e (`rocm-libraries-67us`) and C3f (`rocm-libraries-i190`) are already filed and unblocked by C3d.
