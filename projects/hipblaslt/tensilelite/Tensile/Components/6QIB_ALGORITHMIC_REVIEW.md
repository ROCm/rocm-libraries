# Algorithmic Review — Validator's set-based edge_keys comparison

**Author:** compiler/static-analysis review for `rocm-libraries-6qib`
**Date:** 2026-05-27
**Scope:** evaluate fundamentally-different algorithms vs. the proposed compound-key fix.

---

## Executive verdict

**The current set-based approach is the right shape; compound-key is the right fix; but the team is conflating two separable concerns that should be split first.** Specifically: `edge_keys()` is asked to do *both* canonical-form equality (graph equivalence modulo legal renaming) *and* serve as the trigger for a downstream order-sensitivity classifier (`diagnose_missing_edge` Phase 1). These are different jobs with different invariants, and the dual-requirement "conflict" in `6QIB_DESIGN.md` §1.2 is largely a symptom of conflating them.

Recommended path:
1. **Adopt compound-key (Approach a in 6QIB_DESIGN.md)** as the short-term fix — it is correct, low-risk, and well-scoped.
2. **In parallel, separate the two concerns architecturally**: keep `edge_keys()` as a pure allocation-invariant structural key (byte-key 6-tuple); add a separate explicit `order_check` pass that walks the two graphs *with their node positions* and emits OrderInverted/Reorder failures directly (rather than as a classifier downstream of set-diff). This converts an implicit invariant into an explicit pass and removes the dual-requirement pressure.
3. **Do not adopt graph isomorphism / bisimulation / value numbering** — they all dissolve in this concrete setting because the *same codegen runs on both sides* (per `DEFAULT_SCHEDULER_REFERENCE_DESIGN.md` §1, the SHADOW reference shares writer state with the CMS subject), so node identity is largely shared by construction. Heavy-machinery graph-equivalence is solving a problem we don't actually have.

The investigation chain (n7og → udqg → 32tg → 6qib) converged on a real conflict, but the conflict is between *what edge_keys' set-diff signals* and *what diagnose_missing_edge needs as an input* — not between two genuinely contradictory comparison semantics.

## Current approach in one paragraph

`compare_graphs` (`CMSValidator.py:3578-3744`) computes per-data-flow-category node-count parity (`CMSValidator.py:3671-3709`), then performs a set-difference on `edge_keys()` (`CMSValidator.py:1229-1303`) where each key is a 6-tuple `(producer.identity, consumer.identity, edge_kind, intra_operand_byte_offset, src_operand_slot, sink_operand_slot)`. `identity` is `(canonical_render, emission_ordinal)` with `emission_ordinal` content-stable per `(canonical_render, source_module_id)` pair (`ScheduleCapture.py:778-818`). Each missing key is fed to `diagnose_missing_edge` (`CMSValidator.py:3747+`) which, **as its Phase 1 step (`CMSValidator.py:3819-3850`)**, compares `p_node.position < c_node.position` in the *subject* graph against the same predicate in the *reference* graph and may emit `OrderInvertedFailure`. So the set-diff is acting as a **fault detector** that escalates each disagreement to a per-edge classifier; the classifier — not the keys themselves — is what determines whether the disagreement is a reorder, a missing wait, an SCC clobber, etc.

## The dual requirement framed as a compiler problem

What `edge_keys()` *is asked to do* is graph equivalence modulo a renaming relation R ⊆ Names × Names that is permitted by register-allocator freedom but not by scheduler freedom. In LLVM/GCC terms: this is the *SSA-equivalence under α-renaming* check — but on a pre-register-allocator IR you get it for free, and on a post-register-allocator IR you get it via canonical relabeling. The complication here is that the two captures sit at slightly different points: same SSA-ish operand naming on both sides (because both reference and subject share codegen state per `DEFAULT_SCHEDULER_REFERENCE_DESIGN.md` §3, with the unnamed-Module caveat), but with a residual symbolic-vs-numeric naming asymmetry (`T0_I0` vs `X0_I0`) introduced by the UsePLRPack rotating pack-buffer. The "order sensitivity" requirement is not actually a property of the graph at all — it's a property the diagnostic classifier needs, namely *position-in-stream* information. Position is not part of dataflow-graph equivalence; it's part of the schedule. **The conflict is between "use the graph as a graph" and "use the graph as a schedule," and those are different objects.**

## Candidate alternative approaches

### Approach X1: Graph isomorphism (Nauty/VF2/NetworkX `is_isomorphic`)

**Theoretical fit:** Solves the wrong problem. Isomorphism asks "is there *any* bijection between node sets that preserves edges" — but the validator needs "is the bijection produced by the same-codegen contract structure-preserving," which is a *constrained* isomorphism with already-known node correspondence (identity-tuple equality on the common subset). True isomorphism would mask real bugs that re-permute nodes coincidentally.
**Practical applicability:** Works on `DataflowGraph` shape but requires adapting to handle attributed multi-edges (edge_kind, intra_operand_byte_offset).
**LOC estimate:** ~150–250 (with `networkx` dependency).
**Failure modes:** Binary "is/isn't isomorphic" — useless for `diagnose_missing_edge` which needs to know *which specific edge* differs.
**Scaling:** Worst-case GI is quasi-polynomial (Babai 2016); for sparse attributed graphs of 50–1000 nodes practical complexity is fine. Worse: must reconstruct the per-edge mapping post-hoc.
**Diagnosability:** Poor. The classic complaint about GI-based comparison.
**Verdict:** Dismissed. Solves a problem the validator doesn't have.

### Approach X2: Bisimulation / Hopcroft-Karp partition refinement / Weisfeiler-Leman

**Theoretical fit:** Better than GI for *structural* equivalence — produces canonical labels via successive refinement on neighborhood signatures. Allocation-invariant if labels exclude register names.
**Practical applicability:** Same shape constraint as GI. Would need a custom relabeling pass to handle the multi-edge attribute soup.
**LOC estimate:** ~200–400 if rolled by hand; `networkx` has WL-based canonicalization helpers.
**Failure modes:** When two graphs DON'T match, WL produces refined color classes that differ — somewhat diagnosable but not as direct as "this edge_key is missing."
**Scaling:** O((V+E) log V) per round, k-WL needs k rounds; fine for 50–1000 nodes.
**Diagnosability:** Mid. Color-class diff is interpretable but indirect.
**Verdict:** Dismissed. Same critique as GI — the validator doesn't have an isomorphism problem; it has an *attribute-canonicalization* problem.

### Approach X3: SSA value numbering / GVN-style equivalence

**Theoretical fit:** Excellent in principle. Compute each operand's SSA value number based on producer-instruction + operand role, recursively. Two operands that flow from the same producer get the same number, regardless of physical register. This is *exactly* the canonical form that defeats register renaming.
**Practical applicability:** The validator already has 90% of this — `identity = (canonical_render, emission_ordinal)` IS a value-numbering scheme keyed on rendered text. The breakdown happens when canonical_render embeds register names (per `6QIB_DESIGN.md` §2.1). Switching to pure byte-key + canonical-render-without-operands is essentially a textbook GVN — and is essentially what Approach (a) compound-key does on the edge side.
**LOC estimate:** ~100–200, mostly in identity construction.
**Failure modes:** GVN is hash-based; collisions are possible but the existing identity tuple already deals with them via `emission_ordinal` discriminator.
**Scaling:** Linear in graph size.
**Diagnosability:** Excellent — when two edges have the same value-number key on both sides, they match; when they don't, you can show the user the specific number tuples.
**Verdict:** **Recommended as the principled long-term direction.** Approach (a) compound-key is a partial step toward this.

### Approach X4: Trace equivalence with sliding-window comparison

**Theoretical fit:** Compares emitted instruction sequences with a permutation-tolerant window — addresses "same instructions, different positions" directly. Used in some LLVM scheduler unit tests.
**Practical applicability:** Doesn't address register-naming; would still need GVN-style canonicalization underneath.
**LOC estimate:** ~300–500. Window-size tuning is an empirical knob.
**Failure modes:** False negatives at window boundaries; the test corpus's 11 reorder tests would need each to be re-pinned against a specific window.
**Scaling:** O(N×W) per body; fine.
**Diagnosability:** Good — can point to specific stream positions.
**Verdict:** Dismissed for this layer. Better as a complementary check inside `validate_edge_wait_coverage` (which already does something window-shaped); doesn't replace the graph comparison.

### Approach X5: Tree-edit distance on use-def DAGs

**Verdict:** Dismissed. Tree-edit distance is O(N³) per comparison — bad for the 50–1000 node graphs. Diagnosability is fine but the algorithm gives a *cost*, not a *which-edge*, output.

### Approach X6: Separate-the-concerns refactor (the one I recommend in parallel with a)

**Theoretical fit:** Excellent. `edge_keys()` becomes pure allocation-invariant structural identity (byte-key 6-tuple). A new `compare_schedules(ref_graph, subj_graph)` pass takes both graphs *and their position information* and emits `OrderInvertedFailure` directly by walking the common node set and checking position order — *without going through set-diff at all*.
**Practical applicability:** Surgical. Two function bodies change in `CMSValidator.py`. The 11 reorder/SCC tests need their assertion paths re-routed but the failure objects they assert on are unchanged.
**LOC estimate:** ~50–100 net.
**Failure modes:** The new pass needs the same data the current Phase-1 check uses (`ref_p.position`, `p_node.position`); both are already on `DataflowEdge`/`GraphNode`.
**Scaling:** O(N) extra per comparison — negligible.
**Diagnosability:** Better than today — order failures are reported by the *order pass*, not as a side-effect of *set-diff catching them*.
**Verdict:** **Recommended in parallel with Approach (a)** for the medium term. It's the architectural fix to the conflation noted in `6QIB_DESIGN.md`.

## Critical-question answers

### Q1: Fundamental conflict or solvable by reformulation?

**Solvable by reformulation.** The "conflict" exists only because both requirements are being squeezed through one function (`edge_keys()`) and one comparison operator (set-diff). Allocation-invariance is a *graph-equality* property; order-sensitivity is a *schedule-equality* property. They commute if you compare graphs as graphs AND schedules as schedules — separately. The empirical observation that "byte-key 6-tuple breaks 11 reorder tests" (`6QIB_DESIGN.md` §2.1) is genuinely about the second job being shoehorned through the first. See Approach X6.

### Q2: SSA-style canonicalization possible?

**Yes, and the validator already uses ~90% of it.** `TaggedInstruction.identity_for` (`ScheduleCapture.py:475+`) produces `(canonical_render, emission_ordinal)` where the ordinal is content-stable per source module — that's a value-number. The defect is that `canonical_render` includes register operand names. Stripping operand names from `canonical_render` and supplementing with the byte-key flow (which is already on `DataflowEdge.producer_write_byte_key` / `consumer_read_byte_key`, `CMSValidator.py:1109-1110`, populated but unused in the matching tuple) gives a textbook GVN canonical form. Approach (a) compound-key is a partial application of this idea.

### Q3: Are the 11 reorder tests actually testing edge_keys directly?

**No.** They test `compare_graphs()`'s ability to detect reorders, but the actual order detection lives in `diagnose_missing_edge` Phase 1 (`CMSValidator.py:3819-3850`), which is invoked *after* set-diff finds a missing edge. The mechanism that triggers it under the identity-tuple basis is that the graph builder itself produces a different topology when instructions reorder (e.g., in `test_pack_before_swap_orderinverted`, the producer for v8 at Pack's read time is *LR* in subj vs *Swap* in ref because Swap was moved after Pack — different producer, different edge). Pure byte-key matching still preserves *some* of this discrimination but coarsens it. **This confirms that order-sensitivity is not actually a property of `edge_keys` — it's a property of `diagnose_missing_edge`'s Phase 1, which can be invoked from a different trigger.** Approach X6 makes this explicit.

### Q4: Separation-of-concerns critique

The current architecture conflates: (a) "did SHADOW and CMS produce the same dataflow facts" (graph equivalence), (b) "did CMS reorder a producer past its consumer" (schedule validation against reference), (c) "does CMS have the required waits/barriers" (wait-coverage validation), and (d) "did CMS clobber an SCC value" (intervening-write detection). All four currently run as classifier branches inside `diagnose_missing_edge`, gated by a *missing edge in set-diff*. (b)-(d) don't actually need (a) to fail — they need the per-graph position/wait data, which is independently available. The principled architecture: (a) is its own pass with its own pure structural keys; (b)-(d) are their own passes that walk the graphs directly. The current shape made sense when the only known failure mode was (a) and (b)-(d) were rare classifications, but at the test surface's current density (`6QIB_DESIGN.md` §2.4: hundreds of fixtures, 11 reorder tests), they should be first-class passes.

### Q5: Production-compiler literature

Web search infrastructure was unavailable during this review, so cited from prior knowledge. (a) **LLVM MachineScheduler** uses a DAG node identity that's pointer-equal to the underlying MachineInstr — there's no cross-comparison problem because there's only ever one DAG per region. Its closest analog to this validator's job is `llvm/test/CodeGen/AMDGPU/*.mir` lit tests, which compare *output text* with regex matchers, not graphs. (b) **Translation validation** (Necula 2000, Pnueli/Siegel/Singerman 1998) uses simulation-relation proofs — too heavy for this use case. (c) **Polly / isl** equivalence checking uses Presburger-set equivalence for affine schedules — overkill and inapplicable to the non-affine GPU scheduler. (d) **Alive2** (Lopes et al. 2021) uses SMT-based semantic equivalence — orders of magnitude too heavy. The closest practical analog is (e) the **GCC scheduler regression test methodology**: compare per-pass DAG dumps via canonical relabeling and structural diff — exactly what Approach (a) compound-key + Approach X6 separation would converge on.

### Q6: Smallest change

**Approach (a) compound-key with `ordinal_class = (prod_emission_ordinal, cons_emission_ordinal)`**, leaving everything else untouched. This is the smallest change that resolves the immediate Phase-3 blocker. It preserves order-sensitivity (emission_ordinals differ when instructions reorder), is allocation-invariant (no register names in the key), and reuses existing data. ~20–40 LOC change to `edge_keys()`, no test fixture rework. The architectural separation (Approach X6) is a larger but principled follow-up.

## Recommended approach with concrete rationale

**Adopt Approach (a) — compound-key with `ordinal_class = (producer.emission_ordinal, consumer.emission_ordinal)`** for the immediate fix. File a **new P0 follow-up bead** for the separation-of-concerns refactor (Approach X6) as a dependent of `r62g` Phase 4 cleanup.

Pseudocode for `edge_keys()`:
```python
def edge_keys(self):
    return {
        (e.producer_write_byte_key,         # was: e.producer.identity
         e.consumer_read_byte_key,           # was: e.consumer.identity
         e.edge_kind,
         e.intra_operand_byte_offset,
         e.src_operand_slot,
         e.sink_operand_slot,
         # NEW: ordinal_class — preserves reorder detection trigger
         (e.producer.identity[1], e.consumer.identity[1]))
        for e in self.edges
    }
```

Rationale:
- `producer_write_byte_key` / `consumer_read_byte_key` are already on the edge (`CMSValidator.py:1109-1110`), populated by the resolver — register-rename invariant by construction.
- Including `(producer.emission_ordinal, consumer.emission_ordinal)` preserves the "two physically distinct emissions of the same render get distinct ordinals" property (`ScheduleCapture.py:786-787`) and is content-stable across schedules (per-source-module counter), so a reorder produces a different edge key while a re-naming does not.
- The 30/90 6-tuple structural residuals (`6QIB_DESIGN.md` §2.1 last paragraph) are NOT addressed — they need the granularity-alignment work, which the standing rule requires be filed as a new dependent P0 bead.

## Risks of the recommendation

1. **`ordinal_class` granularity not yet empirically validated.** The 6QIB doc's "(a.1) binary / (a.2) pair / (a.3) stride" enumeration is undecided. Empirically: (a.2) pair is what I recommend above because it preserves the most information without coupling to physical positions; (a.1) binary is too coarse for multi-instruction reorder cycles; (a.3) stride re-introduces position coupling. Phase-3 fixture sweep is required to confirm.
2. **The separation-of-concerns refactor (X6) is correct but requires moving Phase-1 OrderInverted detection out of `diagnose_missing_edge`. Tests that assert on the *failure type* (OrderInvertedFailure vs others) keep working; tests that assert on the *invocation path* (i.e., that Phase 1 fired) would need re-routing.** Audit the 11 reorder tests before committing.
3. **Approach (a) does not fix the 30/90 granularity-asymmetry residuals.** The standing rule applies: file these as a new P0 bead under `r62g` immediately upon landing (a), do not defer.
4. **The SHADOW-vs-CMS reference design (`DEFAULT_SCHEDULER_REFERENCE_DESIGN.md` v5) is the precondition for this analysis.** If Phase 1 nmsx fixes regress, the canonical_render-includes-register-names problem reappears under different cover. Approach (a)'s byte-key basis is robust against this; the identity-tuple basis is not.

## What the team should investigate before committing

1. **Implement Approach (a) with `ordinal_class = (prod_ordinal, cons_ordinal)` and re-run the n7og multi-fixture probe.** Verify: BPG#11 → 0 mismatches, oplb → 0 mismatches, bf16 → 0 mismatches (unchanged), 11 reorder tests still PASS.
2. **Probe the 30/90 6-tuple granularity residuals.** Are SHADOW's per-byte `PackB0` edges and CMS's wide `PackB3` edges semantically equivalent? If yes → file granularity-normalizer bead. If no → file SHADOW capture-pipeline bead. Either way, do not let them mask under compound-key.
3. **Audit the 11 reorder/SCC tests for their actual invocation contract.** Do any assert that `compare_graphs` returns a non-empty failure list (where edge_keys must differ)? Or do they assert on `OrderInvertedFailure` instance shape (where the trigger is irrelevant)? This determines whether X6 can be applied without test-rewrite.
4. **Spike Approach X6** (separation of concerns) on a single test class — e.g., reroute `test_pack_before_swap_orderinverted` to a new `compare_schedules` pass and verify the failure shape is identical. Use this to size the broader refactor.
5. **Document the GVN framing in `EMISSION_ORDINAL_DESIGN.md`** so the principle ("identity = value-number under SSA-like equivalence; ordinal is the discriminator across multi-emissions of one render") survives the next handoff. The current docstring (`ScheduleCapture.py:778-818`) describes the mechanism but not the principle.
