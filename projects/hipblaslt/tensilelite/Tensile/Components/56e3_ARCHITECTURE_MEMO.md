# 56e3 Architecture Memo — Restoring producer-discrimination in byte-key edge matching

**Bead:** `rocm-libraries-56e3` — producer-discrimination loss in byte-key edge matching after C3d
**Status:** architectural review; read-only
**Author:** investigation memo (compiler / static-analysis lens)
**Blocks:** `rocm-libraries-67us` (C3e symmetric direction), `rocm-libraries-si5f` (C3h n7og xfail removal + cross-iter unit tests)

---

## §1 The problem

C3d (`rocm-libraries-xxj4`) made `DataflowGraph.edge_keys()` byte-key based: every edge is now keyed on `(producer_write_byte_key, consumer_read_byte_key, edge_kind, intra_operand_byte_offset, src_operand_slot, sink_operand_slot)`. This is iter-blind and allocation-invariant — pipelined writes across ML iter copies and across UsePLRPack T/X register renames collapse to identical keys, which is the whole point. But it is also **producer-blind**: when two distinct producers A and B write the same byte footprint to the same resource and both feed a consumer C, the edges `A→C` and `B→C` produce *byte-equal* key tuples and cancel under `compare_graphs`' `ref_keys − subj_keys` set-diff. The validator stops surfacing a missing edge in that case even when the wrong producer fed C — and that is precisely the shape of the SCC-clobber, swap-vs-pack reorder, and cross-iter LR0-not-drained tests. 7 tests regressed.

---

## §2 Failure shape — concrete (A, B, C, byte-key) traces

### §2.1 `TestValidateSCCOverlap` (5 of 12 conflict assertions — `test_gr_simple` 2, `test_gr_declaration_order` 2, `test_gr_interval` 3, `test_gr_noshadow` 2, `test_lws` 2, `test_gr_inc_together` 1; 12 total but the bead pins this whole class as the 5-method regression bucket per the failure inventory)

**Default body:** `[Producer:SCmpEQU32 → Consumer:SCSelectB32]`. SCC is a single-bit hardware register; the producer's write-resource is the synthetic SCC sentinel.

**Subject body:** `[Producer:SCmpEQU32 → Clobber:SAddU32 → Consumer:SCSelectB32]`.

| element | producer A | clobber B | consumer C |
|---|---|---|---|
| writes | `('scc', 0)` | `('scc', 0)` | — |
| reads | — | — | `('scc', 0)` |

**REF edge:** `A → C`. `producer_write_byte_key = (('scc', 0),)`, `consumer_read_byte_key = (('scc', 0),)`, `edge_kind='raw_intrawave'`, `intra=(0,)`, `src=0`, `sink=0`.
**SUBJ edge:** `B → C`. *Byte-equal* tuple — A and B both wrote the SCC sentinel, the consumer's read-byte-key is invariant.

`ref_keys − subj_keys = ∅`. Neither `OverriddenInputFailure` nor any other Failure surfaces. The SCC-clobber branch in `diagnose_missing_edge` (`CMSValidator.py:~3897-3934`) is never reached because that branch lives downstream of `compare_graphs` finding a missing key. Pre-C3d, this worked because the *identity* tuples of A and B differed (different `canonical_render`, different `emission_ordinal`), so `A→C` had a different edge_key than `B→C` and the SUBJ edge looked "missing" relative to REF; `diagnose_missing_edge` then matched consumer C and saw a non-A producer with an SCC write between A and C, emitting `OverriddenInputFailure(intervening_writer=B)`.

### §2.2 `test_pack_before_swap_orderinverted`

**Reference:** `[LR → SWait → Swap@2 (writes v8) → VCvtPack@4 (reads v8) → MFMA]`.
**Subject:** `[LR → VCvtPack@1 (reads v8) → SWait → Swap@3 (writes v8) → MFMA]` — the Pack is hoisted BEFORE the Swap so it reads stale data.

**REF edge `Swap → VCvtPack`:** producer A = Swap (writes v8), consumer C = VCvtPack (reads v8). `producer_write_byte_key = (('v', 8),)`, `consumer_read_byte_key = (('v', 8),)`, `intra=(0,)`. (VSwapB32 publishes `dst` at src_slot 0; VCvtPack reads at sink_slot 0 or 1 depending on operand position — either way the slots and byte-key are stable.)

**SUBJ side:** there IS a v8 producer in the stream — but it is the LR (`LRA0` wrote v8 to set up the swap input!). The per-byte `latest_writer[('v', 8)]` at the moment SUBJ's VCvtPack@1 is processed is the LR. So SUBJ emits an edge `B = LR → C = VCvtPack` with `producer_write_byte_key = (('v', 8),)`, identical to REF.

Or, more representative of the pinned bug: even if you ignore the LR-as-producer subtlety, the REF Swap and the SUBJ Swap themselves both write v8 with the same byte-key tuple — and an edge `Swap → MFMA` in REF and `Swap → MFMA` in SUBJ both have key `((v,8), (v,8), ...)`. The Phase-1 `OrderInvertedFailure` branch fires off a *missing* REF key in `diagnose_missing_edge`. Under byte-key keying, the REF edge `Swap → VCvtPack` is *not* missing in SUBJ's edge set — SUBJ has *some* `(v,8)→(v,8)` edge feeding the VCvt (it's just feeding from a different producer). Set-diff empty → Phase 1 never runs → `OrderInvertedFailure` not emitted.

### §2.3 `test_negative_one_prev_iter_lr0_not_drained` (TestGRNotTooEarlyDtlPlusLdsBufGraph)

**Reference ML-1:** `[LRA0@0 (writes v8..v11), LRA0@2 (writes v8..v11 again), SWait, SBarrier]` followed by ML `[GR@2 (reads via LDS), SWait, SBarrier, GR_B]`.
**Subject ML-1:** `[LRA0@0 (writes v8..v11), SWait, SBarrier, LRA0@5 (writes v8..v11) — NOT DRAINED]` followed by an identical ML.

**REF edge:** `LRA0@2 → GR@2` via the `lr_to_gr_lds_reuse` barrier pattern (LR before drain, GR after barrier in next body). Producer A = `LRA0@2`. Both producer- and consumer-byte-keys derive from the LDS-slot pin (`_resolve_dst_resource`), shape `(('v', 8), ('v', 9), ('v', 10), ('v', 11))`.

**SUBJ side:** the misplaced `LRA0@5` lives AFTER the SWait/SBarrier; the `_collect_pattern` walker pairs each LR producer with the *subsequent* `SWait → SBarrier → consumer-of-other-category` chain. With the drain/barrier preceding `LRA0@5`, the `LRA0@5 → GR` edge is *not* collected (the pattern requires producer-before-wait-before-barrier-before-consumer). But the FIRST LR `LRA0@0` IS drained and the pattern WILL emit `LRA0@0 → GR`. So SUBJ has ONE `lr_to_gr_lds_reuse` edge with key `((('v',8)..('v',11)), (consumer_byte_keys), 'lr_to_gr_lds_reuse', (0,1,2,3), 0, 0)`.

REF *also* has ONE `LRA0 → GR` edge of the same shape (the LR-to-GR walker has its own pattern semantics and doesn't double-collect across multiple-LR-producers in a single window; even if it did, both REF LRs would produce byte-equal keys to SUBJ's surviving LR).

The two key sets are byte-equal. `ref_keys − subj_keys = ∅`. The missing `LRA0@5 → GR` edge does not surface; the validator misses the "LR not drained before next-iter GR" defect.

### §2.4 Common shape

In all three failure clusters, the validator's view of the residual after set-diff is empty because two byte-equal-but-producer-distinct edges cancel:

| test | byte-key | A (REF producer) | B (SUBJ wrong producer) | cancellation |
|---|---|---|---|---|
| SCCOverlap | `('scc', 0)` | the legitimate SCC writer (e.g. SCmpEQU32) | the clobber (e.g. SAddU32) | both feed the same SCC reader |
| pack_before_swap | `('v', 8)` | Swap@2 (legitimate) | LR producer or *the same Swap in a different stream position* | both feed the same VCvtPack |
| prev_iter_lr0_not_drained | `(('v',8)..('v',11))` LDS-pin | LRA0@2 (drained) | LRA0@0 (drained — the misplaced LRA0@5's edge never forms) | both feed the same next-iter GR |

The cases differ in *why* there are two producers (a real clobber, a real reorder of a write, a real LR-displacement that prevents the bad edge from forming at all), but they share the cancellation mechanic.

---

## §3 Option A — Hybrid key with producer-ordering rank

**Shape:** `(producer_rank, prod_byte_key, cons_byte_key, edge_kind, intra, src_slot, sink_slot)`, where `producer_rank = index of this edge's producer in subj_graph.byte_key_writers[prod_byte_key]` (i.e. "this producer is the Nth writer to this byte-key, ordered by unrolled_position").

**Does it resurface the 5 failing tests?**
- SCCOverlap: REF has 1 writer to `('scc', 0)` (rank 0). SUBJ has 2 writers (A rank 0, B rank 1). The REF edge has rank 0 and SUBJ has both a rank-0 edge (A → C) AND a rank-1 edge (B → C) — but C only reads SCC once, so SUBJ only emits ONE edge per consumer-read. The SUBJ edge is `B → C` with rank=1 (B is the latest writer at consumer time). REF edge has rank=0. **Different keys → set-diff exposes the missing rank-0 edge → `OrderInvertedFailure`/`OverriddenInputFailure` fires.** Works.
- pack_before_swap: REF has Swap as rank-0 writer of v8 (only writer). SUBJ has LR-then-Swap; Swap is rank-1. SUBJ edge to VCvtPack@1 sources from LR (rank-0), REF edge sources from Swap (rank-0 in REF). Same numeric rank. Hmm — for this case it depends on what the writer history looks like on each side: REF v8 writers = [Swap@2], SUBJ v8 writers = [LR@0, Swap@3]. So REF's only-writer is Swap rank 0; SUBJ's last writer at VCvtPack@1 is LR rank 0. Both rank 0 → cancels still. **Fails for this case.**
- prev_iter_lr0_not_drained: REF writers to the LDS byte-key = [LRA0@0, LRA0@2] (both drained). SUBJ writers = [LRA0@0, LRA0@5] (only @0 drained). The barrier-pattern emits one edge per producer that satisfies the pattern. REF emits 2 edges (ranks 0, 1). SUBJ emits 1 edge (rank 0). The rank-1 edge is missing in SUBJ → set-diff exposes it. **Works.**

**Identity iter-blindness contract (C3a):** ranks are computed from unrolled_position order within a single graph. ML iter copies of the same write would each occupy a separate rank slot (so REF has [iter0-write@pos50, iter1-write@pos170], ranks 0 and 1; SUBJ has the same two writes at maybe different positions but same ordering, ranks 0 and 1). For symmetric pipelining cases, the ranks line up. **Preserved.**

**n7og pipelining cancellation:** the BPG#11 PackB3 writes appear at ML[K] and ML[K+1] in both SHADOW and CMS unrolled streams. Both sides have the same writer-history for the relevant byte-keys; the consumer's edge points to the same rank on both sides → cancels. **Preserved IF the unrolled walk indeed produces the same writer-history on both sides.** That's the C3c contract.

**Risk:** false positives when the producer rank-history shifts asymmetrically across REF and SUBJ even though the dataflow is correct. Example: REF writers to `('v', N)` = [W1@pos50, W2@pos170, W2@pos300] (W2 writes twice); SUBJ has [W2@pos50, W1@pos170, W2@pos300] (legitimate reorder where W1 and W2 swap positions but the consumer still ends up reading the right value because the final-rank writer is the same W2). If consumer reads at pos400, both sides see W2 as the final writer. But the *rank* of W2 on REF is 2 (third writer) and on SUBJ is 2 (third writer) — that survives. The pathological case is a writer-history of unequal LENGTH on the two sides. That happens precisely when CMS deletes or inserts writes — and that's a real defect we want to surface, so it's not a false positive.

The pack_before_swap failure shows the limitation: when SUBJ's wrong-producer history is shorter than REF's (LR alone vs Swap alone, ranks 0 vs 0), ranks cannot discriminate.

**Implementation cost (semantic):** edit `DataflowEdge` to carry rank or compute it at `edge_keys()` time from `byte_key_writers`; rebuild ref_edges_by_key with rank included; document the rank semantics.

**Future implications (C3e, C3f):** C3e adds the symmetric `subj_keys − ref_keys` direction; rank addition is symmetric so this is fine. C3f migrates Phase 0/1 to byte-key reverse-index lookup; rank gives that lookup a natural disambiguator.

**Verdict:** Partial fix — handles SCC clobber and the LR-not-drained case but NOT the pack-before-swap case where REF and SUBJ each have a single (different) producer for the same byte-key. Insufficient as a standalone solution.

---

## §4 Option B — Two-phase compare_graphs: byte-key narrowing + per-position ordering check

**Shape:** Keep byte-key set-diff as Phase 1 (catches missing/extra edges by physical byte footprint, the C3d contract). Add a Phase 2 over byte-keys present on BOTH sides: for each `(bk, consumer_identity)` pair, compare the producer-identity that REF assigned vs the producer-identity that SUBJ assigned. Emit a new `WriterMismatchFailure` (or repurpose `OverriddenInputFailure` more broadly) when they differ.

**Does it resurface the 5 failing tests?**
- SCCOverlap: Phase 1 cancels. Phase 2 walks `('scc', 0)` consumers; for C = consumer, REF producer-identity = A's identity, SUBJ producer-identity = B's identity → mismatch → emit `OverriddenInputFailure(intervening_writer=B)`. **Works.**
- pack_before_swap: Phase 1 cancels (REF edge for `Swap→VCvtPack`, SUBJ edge for `LR→VCvtPack` — but actually under the per-byte resolver SUBJ's VCvtPack@1 sources from LR@0). Phase 2 finds: for consumer VCvtPack(identity X), REF producer = Swap (identity Y), SUBJ producer = LR (identity Z). Mismatch → emit a routing-divergence failure. The pinned assertion wants `OrderInvertedFailure` specifically, so we'd need either (a) repurpose the same Failure type or (b) update the test to accept the new typed failure. **Works on the semantic level; may need test re-pinning.**
- prev_iter_lr0_not_drained: Phase 1 cancels (both sides emit one LR-to-GR edge with the same byte-key). Phase 2: for consumer GR, REF producer = LRA0@2, SUBJ producer = LRA0@0. Different producer identities → emit failure. The test asserts `MissingWaitFailure` specifically, which is a Phase-2 (in the *original* sense) outcome that walks SWait coverage. Under Option B, we'd surface the *fact* that the producer history differs, but routing through to `MissingWaitFailure` requires the new check to invoke `validate_edge_wait_coverage`-style logic on the divergent edge. Doable but adds complexity. **Works at the semantic level; the failure-type pinning may need rethinking.**

**Identity iter-blindness contract:** Phase 2 uses producer identity for comparison. Identity is iter-blind (`(canonical_render, source_module_id, emission_ordinal)` — same across iter copies of the same source instruction). Two iter copies of the same instruction have the same identity, so the same-identity-on-both-sides cancellation persists. **Preserved.**

**n7og pipelining cancellation:** Phase 1 still cancels n7og's pipelined edges (the whole point of C3d). Phase 2 runs only over byte-keys present on both sides; for n7og's pipelined writes, REF and SUBJ both produce the same producer identity (same source instruction in different iter copies → same identity by construction). **Preserved.**

**Risk:** new failure mode `WriterMismatchFailure` overlaps with `OrderInvertedFailure` and `OverriddenInputFailure` in scope; need to define a clean classification. New code path; new diagnostic shape. Could potentially produce false positives if REF and SUBJ legitimately have a routing difference that is allowed (e.g. CMS legitimately reorders writes when the resulting value is identical).

**Implementation cost (semantic):** introduce Phase 2 inside `compare_graphs`; iterate `ref.byte_key_writers ∩ subj.byte_key_writers`; for each byte-key, walk consumers and compare the writer-identity-at-consumer-time on each side; emit appropriate Failures. Requires building a `byte_key → consumers` index (the consumer side of `byte_key_writers`, currently absent).

**Future implications (C3e, C3f):** C3e (symmetric direction) naturally folds into Phase 2 — the comparison is already two-sided. C3f (Phase 0/1 migration to byte-key reverse-index) is *aided* by this — Phase 2's writer-identity comparison IS the byte-key reverse-index query C3f wants to install.

**Verdict:** Architecturally cleanest. Cleanly separates "what edges should exist by byte footprint" from "which producer actually fed which consumer". The two questions are genuinely different and deserve different phases. Adds code but the new code is principled and aligned with the C3e/C3f roadmap.

---

## §5 Option C — Augment edge_keys() with the producer's identity 3-tuple

**Shape:** `(producer.identity, producer_write_byte_key, consumer_read_byte_key, edge_kind, intra, src_slot, sink_slot)` where `producer.identity = (canonical_render, source_module_id, emission_ordinal)` is iter-blind.

**Was identity dropped in C3d for principled reasons or to enable cleanup?** Per `C3d_xxj4_IMPL_PLAN.md` §G ("Identity references to remove"), C3d's intent was that **no** identity references remain in the keying basis — the docstring rewrite explicitly says "no identity-tuple references remain in the keying basis." The motivation was that `canonical_render` embeds register operand names, making identity allocation-sensitive (T0_I0 vs X0_I0 in UsePLRPack). However, per `UNROLLED_VALIDATION_PLAN.md` §2.1 "Identity stays `(canonical_render, source_module_id, emission_ordinal)` — iter-blind, exactly as today." Identity is iter-blind because the same instruction in iter 0 and iter 1 has the same source instance → same `(canonical_render, source_module_id, emission_ordinal)`.

The allocation-sensitivity concern (T0/X0) is real: under UsePLRPack the same physical register is referenced by different symbolic names across iter copies, so `canonical_render` of writes that touch the same physical register can differ between REF and SUBJ. This would re-introduce false positives for n7og.

**Does it resurface the 5 failing tests?**
- SCCOverlap: A and B have different identities (different `canonical_render`: `SCmpEQU32 vs SAddU32`). REF edge has A's identity, SUBJ edge has B's identity → different keys → set-diff exposes. **Works.**
- pack_before_swap: REF edge `Swap → VCvtPack` has Swap's identity; SUBJ edge `LR → VCvtPack` (or whatever wrong producer SUBJ resolves to) has LR's identity → different → exposes. **Works.**
- prev_iter_lr0_not_drained: REF edge has LRA0@2's identity (per identity rules, `emission_ordinal` differentiates the two LRA0 instances even if their `canonical_render` is identical), SUBJ has LRA0@0's. Different → exposes. **Works.**

**Identity iter-blindness contract:** iter copies share identity → keys cancel across iter copies of the same source instruction. **Preserved.**

**n7og pipelining cancellation:** PROBLEMATIC. The whole reason C3d dropped identity was that under UsePLRPack the canonical_render of pipelined Pack instructions embeds different register names (T0_I0 in SHADOW vs X0_I0 in CMS for the same physical register). Adding identity back re-introduces those false positives. Per `n7og_PROBE_REPORT.md` the SHADOW pipeline's name_to_idx issue and the T/X register-renaming asymmetry are real and would re-fire under Option C unless we also re-engineer `canonical_render` to strip operand names. **NOT preserved as currently designed.**

The fix shape would be: define a new "structural identity" that excludes `canonical_render` and uses only `(source_module_id, emission_ordinal)`. That IS iter-blind AND register-name-blind. It uniquely identifies a source instruction without embedding its operand spellings. Call it `source_identity`.

**Refined Option C′:** Use `source_identity = (source_module_id, emission_ordinal)` in the key tuple, NOT `canonical_render`.

- Does it discriminate A vs B in SCCOverlap? Yes — different source instructions have different `(source_module_id, emission_ordinal)` regardless of operand spelling.
- Does it preserve iter-blindness? Yes — iter copies share `source_module_id` and `emission_ordinal` (they wrap the same source `TaggedInstruction`).
- Does it preserve n7og cancellation? Yes — the SAME source instruction in SHADOW and CMS captures has the same `source_module_id` and `emission_ordinal`; the T/X register name lives in `canonical_render` which we no longer key on.

**Implementation cost (semantic):** add a `source_identity` field on `GraphNode` (or compute it inline from existing fields), thread it into `edge_keys()` and `ref_edges_by_key`, update docstrings. Modest.

**Future implications (C3e, C3f):** Symmetric direction is fine. Phase 0/1 byte-key reverse-index lookup in C3f still works — `source_identity` is an *additional* discriminator on the edge key, not on the lookup index.

**Risk of Option C′:** does C3a's iter-blind identity guarantee actually hold? Per `EMISSION_ORDINAL_DESIGN.md` and the n7og report's A3 finding ("Identities match (the 3-tuple is byte-equal). Category and slot DIFFER"), yes — iter copies of the same source instruction yield byte-equal identity tuples by construction. Even the 3-tuple form is iter-blind because emission_ordinal is computed from the source module's emission stream, not from the iter copy index.

There is one subtlety: `source_module_id` for the prologue / NGL / NLL bodies is distinct from ML's `source_module_id`, but that's correct — instructions in different bodies are genuinely different source instructions. For ML iter copies, `source_module_id` is the SAME (they all derive from the same ML source module).

**Verdict:** Option C as originally stated is wrong (re-introduces UsePLRPack false positives). Option C′ with `source_identity = (source_module_id, emission_ordinal)` (no `canonical_render`) is principled, surgical, and aligned with the C3a iter-blindness contract. Strong candidate.

---

## §6 Option D — Use byte_key_writers reverse index in compare_graphs directly

**Shape:** Don't change `edge_keys()`. Change `compare_graphs` to consult `byte_key_writers` directly: for every byte-key in either graph's edge set, compare the writer-history-at-each-consumer between REF and SUBJ.

**Does it resurface the 5 failing tests?** Mechanically identical to Option B — same byte-key reverse-index comparison, just packaged differently. Yes for SCC, pack_before_swap, prev_iter (with the same caveats about failure-type pinning).

**Identity iter-blindness contract:** preserved (uses identity for the writer comparison, but identity is already iter-blind).

**n7og pipelining cancellation:** preserved.

**Risk:** abandons the clean set-diff abstraction. `compare_graphs` becomes a procedural walk over byte-keys and consumers rather than a tuple-equality set diff. Harder to read; harder to extend; harder to compose with C3e symmetric direction.

**Implementation cost (semantic):** rewrites the core of `compare_graphs`. Higher than Option B.

**Future implications:** This is *almost* what C3f proposes for `diagnose_missing_edge` Phase 0/1 — but the C3f scope keeps `compare_graphs` itself byte-key set-diff based and migrates the *post-residual* phases. Option D collapses both layers into one, losing the staging that makes diagnose_missing_edge inspectable.

**Verdict:** Strictly worse than Option B — same semantic correctness, less compositional cleanliness, harder to test. Reject.

---

## §7 Option E — Augment edge_keys() with producer's `source_identity` (Option C′ as standalone)

Already covered in §5; written here as the standalone proposal.

**Shape:** `(source_identity, producer_write_byte_key, consumer_read_byte_key, edge_kind, intra, src_slot, sink_slot)` where `source_identity = (source_module_id, emission_ordinal)`.

**Discriminates producers** in all 5 failing tests (different source instructions have different `source_identity`).
**Iter-blind** (iter copies share source).
**Register-rename-blind** (no `canonical_render` in the key).
**Single-stage**: still a set-diff. No new compare_graphs phase. The set-diff residual still feeds `diagnose_missing_edge`, which already has `OverriddenInputFailure`, `OrderInvertedFailure`, `MissingWaitFailure` classification.

**Failure-type pinning:** under Option E, when SCC A→C cancels-then-doesn't, the REF edge `A→C` is missing from SUBJ keys; `diagnose_missing_edge` runs on the REF edge, walks SCC handling, finds B as the intervening SCC writer in SUBJ, emits `OverriddenInputFailure(intervening_writer=B)`. **Test-shape compatible without re-pinning.** Same for `OrderInvertedFailure` (Phase 1 sees REF positions vs SUBJ positions of the same source instructions) and `MissingWaitFailure` (Phase 2 walks SWait coverage on the missing edge). All three failing-test classes resurface their *exact* original Failure types.

---

## §8 Comparison table

| Question | A (rank) | B (two-phase) | C (full identity) | C′/E (source_identity) | D (reverse-index in compare) |
|---|---|---|---|---|---|
| Resurfaces 5 failing tests? | partial (fails pack_before_swap) | yes (may need failure-type re-pinning) | yes | **yes (no re-pinning)** | yes (failure-type re-pinning) |
| Preserves C3a iter-blindness? | yes | yes | yes (identity is iter-blind) | **yes** | yes |
| Preserves n7og pipelining cancellation? | yes (if writer-history symmetric) | yes | **NO** (re-introduces T/X false positives) | **yes** | yes |
| False-positive risk | moderate (writer-history asymmetry under legitimate reorders) | low (Phase 2 only fires on real divergence) | high (UsePLRPack regression) | **low** | low |
| Implementation cost (semantic scope) | touches DataflowEdge, edge_keys, ref_edges_by_key | touches compare_graphs (new Phase 2), needs consumers-by-byte-key index | touches edge_keys, ref_edges_by_key only | **touches edge_keys, ref_edges_by_key, plus source_identity helper on GraphNode** | rewrites compare_graphs core |
| C3e (symmetric direction) interaction | symmetric, fine | folds into Phase 2 naturally | symmetric, fine | **symmetric, fine** | symmetric, fine |
| C3f (Phase 0/1 byte-key reverse-index) interaction | rank gives natural disambiguator | Phase 2 IS the migration target | independent | **independent** | collapses compare/diagnose layers |

---

## §9 Recommendation

**Adopt Option E (Option C′) — augment `edge_keys()` with `source_identity = (source_module_id, emission_ordinal)`.**

Justification:

1. **It is the smallest principled change that restores producer-discrimination without re-introducing the allocation-sensitivity C3d was designed to eliminate.** The key tuple grows by one iter-blind, register-rename-blind discriminator. Set-diff abstraction stays intact; `diagnose_missing_edge` keeps its existing branching and emits the *exact* Failure types the failing tests pin (no re-pinning).

2. **It is consistent with the original C3a iter-blindness contract.** Per `EMISSION_ORDINAL_DESIGN.md` and the n7og report's empirical finding, iter copies of the same source instruction share `source_module_id` and `emission_ordinal`. The 3-tuple `(canonical_render, source_module_id, emission_ordinal)` is the IDENTITY contract; the 2-tuple `(source_module_id, emission_ordinal)` is the structural-identity subset that drops the allocation-sensitive operand spelling.

3. **It does not re-introduce the UsePLRPack/n7og false-positive regression.** Option C as literally stated in the prompt (full 3-tuple identity) WOULD re-introduce it — that's why C3d removed identity. The refinement is critical: key on the structural identity subset that excludes `canonical_render`. The probe report (n7og §A3) explicitly confirms iter copies under UsePLRPack share `(source_module_id, emission_ordinal)` even when the rendered canonical text differs.

4. **It composes cleanly with the C3e/C3f roadmap.** Symmetric direction is a per-direction set-diff; structural identity is symmetric. Phase 0/1 migration to byte-key reverse-index is orthogonal — the reverse-index is still keyed on byte-keys; the edge-discriminator just gains one more field.

5. **Option B is the closest competitor but pays for cleanliness with a new failure-classification surface.** Adopting B is the right call IF we want to elevate "writer-mismatch" to its own named typed failure (with its own diagnostic format). That's a reasonable design choice — but it is a SECOND option-space change on top of C3d, layered over an already-large transition. Option E is one focused change that resurfaces the existing Failure taxonomy.

6. **Option A is structurally insufficient** (fails pack_before_swap when REF and SUBJ each have a single, distinct producer for the same byte-key).

7. **Option D collapses abstractions** (compare and diagnose lose their staging).

8. **Option C as literally stated is wrong** — but Option C′/E is what the prompt was really pointing at.

---

## §10 Failure mode + verification probe

**What could go wrong with Option E:**

1. **`source_module_id` collision across PRO/ML/NGL/NLL bodies could be incomplete.** If two genuinely distinct source instructions in different bodies share the same `(source_module_id, emission_ordinal)`, they would collide. The n7og probe §A3 shows shared rocisa instances ACROSS bodies (40 of 64 CVT instances in NLL share `id(rocisa_inst)` between SHADOW and CMS) but those are the SAME source instruction — collision is correct, not a bug. The risk is in the opposite direction: two distinct source instructions assigned the same emission_ordinal due to a stamping bug.

2. **Iter-blindness under `UnrolledCapture` materialization.** Under C3a, ML iter copies are explicitly stamped with the same `identity` but distinct `unrolled_position`. The `source_module_id` and `emission_ordinal` come from the underlying `TaggedInstruction`, which is shared across iter copies. Need to verify that materialization preserves this. If a future change to `UnrolledCapture.from_four_part_capture` deep-copies the TaggedInstruction (changing `source_module_id`), iter-blindness silently breaks.

3. **Cross-body byte-key collisions in edge_kind=`raw_intrawave`.** Two different source instructions in different bodies that happen to write the same byte-key (e.g. both write `('s', 50)` at different times) — under E, the edge keys differ by `source_identity` even if the byte-key matches. That's correct: they ARE different edges. No false negative.

**Verification probe (concrete tests to add):**

1. **Regression test on the 5 failing test classes.** All 5 must surface their original Failure types (`OverriddenInputFailure` for SCC, `OrderInvertedFailure` for pack_before_swap, `MissingWaitFailure` for prev_iter_lr0_not_drained). Pin the Failure shape exactly as today.

2. **n7og non-regression**: The bf16 fixture in `test_n7og_edge_keys_multifixture.py` must remain at 0 mismatches. The two TF32 xfails (gated on `udqg`) must STAY at their current XFAIL count (Option E does not change the SHADOW pipeline's broken name_to_idx; the byte-keys still misalign).

3. **Synthetic positive test for iter-blindness under E.** Construct a tiny capture with the SAME source instruction in 2 ML iter copies writing the same byte-key, feeding a consumer in NGL. Verify that REF and SUBJ (with different scheduling of the ML→NGL pipelining) produce byte-equal `edge_keys()` for the resulting cross-iter edge.

4. **Synthetic positive test for register-rename-blindness under E.** Construct a tiny capture where the SAME source instruction renders differently in SHADOW vs CMS (e.g. symbolic vs numeric operand). Verify that `source_identity` is byte-equal across the two captures even when `canonical_render` differs.

5. **Bead acceptance gate**: re-run `test_ValidateSCCoverlap.py`, `test_validate_pack_graph.py::TestSwapPackGraph::test_pack_before_swap_orderinverted`, `test_validate_gr_not_too_early_graph.py::TestGRNotTooEarlyDtlPlusLdsBufGraph::test_negative_one_prev_iter_lr0_not_drained` to clean pass; verify no NEW failures in the full unit suite.

---

## §11 Implementation outline (semantic scope)

1. **Define `source_identity` accessor** on `GraphNode`. Two design choices:
   - A new property `GraphNode.source_identity` returning `(self.identity[1], self.identity[2])` (the slice of the existing identity tuple).
   - A computed property reading directly from the underlying `TaggedInstruction` / `source_module_id` / `emission_ordinal` fields. Less coupling to the identity tuple's exact shape.
   The first is simpler; the second is more future-proof. Pick whichever aligns with current property conventions on `GraphNode`.

2. **Augment `DataflowGraph.edge_keys()`** to include `producer.source_identity` as the first element of the tuple. Update docstring to explain why we key on producer source identity (producer discrimination) AND byte-key (allocation invariance, iter-blind across non-source dimensions).

3. **Mirror in `compare_graphs.ref_edges_by_key` construction** — exact same tuple shape.

4. **Update `diagnose_missing_edge` and `_classify_edge_coverage` docstrings** that reference the keying basis. No logic change in those functions; they consume edges via the `ref_edge` parameter, not by reconstructing keys.

5. **Update `CMSValidator.py:3645-3713` block comment** (the long history of identity → byte-key migration). Add an entry: "56e3 refinement: producer `source_identity` re-introduced into the key to restore producer-discrimination without re-introducing the canonical_render-driven allocation sensitivity that motivated C3d."

6. **Delete the `56e3` known-limitation paragraph** in `compare_graphs` docstring (`CMSValidator.py:3645-3651`) — it documents a limitation Option E closes.

7. **Tests**: as enumerated in §10. The pinned 5 failing tests will once again fail-as-expected (i.e. surface the pinned Failure types on the misordered subject capture). No test re-pinning needed.

8. **C3e and C3f sequencing**: after Option E lands, C3e (symmetric direction) and C3f (Phase 0/1 byte-key reverse-index) proceed unblocked. The reverse-index lookup in C3f can use `byte_key_writers` directly (no change needed); the Phase 1 order check naturally uses `unrolled_position` (already on edges as diagnostic annotations).

---

**End of memo.**
