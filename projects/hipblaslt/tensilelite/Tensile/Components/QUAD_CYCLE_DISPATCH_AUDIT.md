# Quad-cycle dispatch audit (rocm-libraries-o0ei)

Investigation memo for the order-dependent dispatch in
`_classify_edge_coverage` and the duplicate dispatch in
`diagnose_missing_edge`. Audits three (plus one) refactor options,
compares against the in-flight `rocm-libraries-009` rocisa-category
work, and recommends a path forward.

> Status: investigation only in this commit. Implementation is
> deferred to a follow-up bead — see §8.

---

## 1. Summary

`_classify_edge_coverage` (`CMSValidator.py:2916`) dispatches each
producer→consumer edge to one of four typed end-states (PackMFMA→CVT
gap, generic MFMA gap, CVT→MFMA gap, ALU exemption) via four sequential
`if` branches. The branches share **overlapping predicates** —
PackMFMAs satisfy `_is_mfma_producer`, and CVTPacks satisfy
`_is_alu_producer` — so the chain MUST be checked in a specific order:
PackMFMA→CVT first, then generic MFMA, then CVT→MFMA, then ALU. Any
swap silently downgrades a strict gap (5 quad-cycles for PackMFMA→CVT;
2 quad-cycles for CVT→MFMA) to either the loose generic finish-cycle
gap or the ALU pass-through, and the validator stops emitting
`TimingTooCloseFailure` for the offending edges. `diagnose_missing_edge`
(`CMSValidator.py:2508`) carries an **identical** dispatch chain at
`:2670`, `:2691`, `:2712`, `:2728`, doubling the surface area for the
same fragility.

**Top recommendation:** Option 1 (single dispatch table) is the
smallest, most local, and most order-independent fix. It can land in
~80–100 LoC, eliminates the order dependency by construction (a
`dict[(producer_kind, consumer_kind), helper]` has no order to begin
with), and de-duplicates the two copies in `_classify_edge_coverage`
and `diagnose_missing_edge` into one table. **However** —
`rocm-libraries-009` (rocisa semantic-category attribute, in flight,
status `in_progress`, P1) will collapse the predicate chain
`_is_mfma_producer` / `_is_cvt_pack_producer` / `_is_alu_producer` to
one-line attribute reads. Once `009` lands the dispatch table becomes
even cheaper to express (the keys are literal category enum values
rather than predicate-call results), AND a fifth option opens up:
classify the producer kind ONCE per edge into an enum and dispatch by
exact-match lookup, with no overlap to disambiguate. **Recommendation:
file a follow-up bead. Defer dispatch-table refactor until after
`009`.** Reasoning in §8.

---

## 2. Current shape

### 2.1 The four end-states

`_classify_edge_coverage` (`CMSValidator.py:2916`) classifies every
edge into exactly one of:

| End-state             | Predicate(s)                                                     | Helper                       | Threshold              |
|-----------------------|------------------------------------------------------------------|------------------------------|------------------------|
| PackMFMA → CVTPack    | `_is_mfma_pack_producer(p)` AND `_is_cvt_pack_producer(c)`       | `_mfma_pack_to_cvt_gap_ok`   | 5 quad-cycles          |
| MFMA-as-producer      | `_is_mfma_producer(p)` (= `category=="MFMA"` OR PackMFMA)        | `_quad_cycle_gap_ok`         | finish-cycles for inst |
| CVTPack → MFMA        | `_is_cvt_pack_producer(p)` AND `_is_mfma_producer(c)`            | `_cvt_to_mfma_gap_ok`        | 2 quad-cycles          |
| ALU-immediate         | `_is_alu_producer(p)`                                            | (none — early return `[]`)   | n/a                    |
| (fall-through)        | none of the above                                                | Phase-2 wait coverage        | n/a                    |

Each end-state is a typed early-return; nothing falls through after a
match. The fall-through path is the wait-coverage classifier.

### 2.2 The load-bearing branch order

The branches share two-way overlaps that make ordering load-bearing.
Code refs from `_classify_edge_coverage`:

- **PackMFMA→CVT vs. generic MFMA** (`CMSValidator.py:2952` vs.
  `:2972`). `_is_mfma_pack_producer(p_node)` returns True for the
  TF32 4x4 PackMFMA pattern (`category.startswith("Pack")` AND
  rocisa class is `MFMAInstruction`). `_is_mfma_producer` ALSO
  returns True for those producers — its definition at
  `CMSValidator.py:1442` literally calls
  `_is_mfma_pack_producer` on the `category != "MFMA"` branch. So a
  PackMFMA→CVTPack edge satisfies BOTH predicates. If the generic
  MFMA branch ran first, it would route the edge through
  `_quad_cycle_gap_ok`, whose threshold is the producer's
  `_mfma_finish_cycles_for(...)` — `1` for the 4x4 PackMFMA — and
  would silently pass an edge that the 5-cycle settle window would
  flag.

- **CVTPack→MFMA vs. ALU exemption** (`CMSValidator.py:2997` vs.
  `:3011`). `_is_cvt_pack_producer(p_node)` returns True for
  `Pack*`-categorized producers whose rocisa class is the
  `VCvtPkF32toBF16` family. `_is_alu_producer` returns True for
  EVERY `Pack*`-categorized producer except the PackMFMA carve-out
  (`CMSValidator.py:1913–1918`). So a CVTPack producer satisfies
  BOTH predicates. If the ALU exemption ran first, it would
  silently pass the edge with no gap check, and the 2-quad-cycle
  CVT→MFMA settle window would never be enforced.

- **PackMFMA→CVT vs. ALU exemption** (`CMSValidator.py:2952` vs.
  `:3011`). PackMFMA producers do NOT satisfy `_is_alu_producer`
  (the explicit carve-out at `CMSValidator.py:1917–1918` returns
  False), so this pair has no ordering constraint. The constraint
  is only between PackMFMA→CVT and the generic MFMA branch above.

The author has documented each carve-out in-line (search for "Must
run BEFORE" comments at `:2942-2947` and `:2987-2996`); the
documentation is correct but the dispatch pattern is what the bead
calls a "design smell."

### 2.3 Duplicated dispatch in `diagnose_missing_edge`

`diagnose_missing_edge` (`CMSValidator.py:2508`) carries the same
four-branch dispatch chain at lines `:2670`, `:2691`, `:2712`,
`:2728`, in the same order, with the same in-line "Must run BEFORE"
comments. The two callers do nearly identical work — they construct
the same `TimingTooCloseFailure` from the same `TimingCheck` shape —
but `diagnose_missing_edge` constructs labels eagerly inside each
branch (`cms_node_label(p_node, _body_for_node(...))` repeated four
times) while `_classify_edge_coverage` hoists the labels to the top
(`p_label = cms_node_label(...)` at `:2948`). Functionally identical;
syntactically two separate copies. Any refactor that moves the
dispatch must update BOTH.

### 2.4 Test surface

`Tensile/Tests/unit/test_dataflow_graph_register_gaps.py` imports
`_quad_cycle_gap_ok`, `_cvt_to_mfma_gap_ok`, and
`_mfma_pack_to_cvt_gap_ok` directly (89 references) and tests them
as standalone functions. Any refactor MUST keep the three helpers
callable with their current `(producer, consumer, ...) -> TimingCheck`
signature — collapsing them into a single dispatch entry point is not
free. The dispatch refactor is therefore strictly about HOW
`_classify_edge_coverage` and `diagnose_missing_edge` SELECT among
the helpers, not about replacing the helpers themselves.

Smaller test-side imports: `test_validate_gr_not_too_early_graph.py`
(7 refs), `test_arch_profile_unregistered_isa.py` (12 refs),
`test_ValidateGRsCompleteBeforeLr1s.py` (2 refs). These are gap-helper
unit tests, not dispatch tests; they are unaffected by a dispatch-only
refactor.

### 2.5 What goes wrong if the order breaks

Concrete failure modes a future refactor might silently introduce:

1. **Reorder PackMFMA→CVT after generic MFMA.** PackMFMA→CVTPack edges
   on a TF32 4x4 emulation path get the 1-cycle finish gap instead of
   the 5-cycle settle gap. Any TF32 schedule with a PackMFMA writing
   to a CVT1 4 cycles later will pass the validator even though the
   accumulator hasn't settled.
2. **Reorder CVT→MFMA after ALU exemption.** CVTPack→MFMA edges get
   the no-op ALU pass instead of the 2-cycle settle gap. CMS
   schedules emitting a CVTPack 1 cycle before the MFMA that consumes
   it will pass the validator even though the converted bf16 value
   hasn't propagated to the MFMA's read.
3. **Reorder generic MFMA after CVT→MFMA.** CVT→MFMA still fires
   first (its predicate requires CVTPack producer, not MFMA), so this
   particular swap is safe. The generic MFMA branch's only ordering
   constraint is PackMFMA→CVT-before-it. A regression test would not
   catch the 1↔2 swap directly; only the inversions in (1) and (2)
   above are observable.

The first two are the failure modes the bead refers to as "every
refactor in this area has tripped on it."

---

## 3. Option 1 — Single dispatch table

### 3.1 Sketch

```python
# Module-level table; populated once, queried per edge.
# Key: (producer_predicate, consumer_predicate) — both are predicates
# on (GraphNode), so the table key is the conjunction of the two
# discriminators. Predicates are evaluated in registration order ONLY
# WITHIN a single key entry; across keys, the table guarantees no
# overlap because the registered keys are pairwise non-overlapping by
# construction (PackMFMA carve-out vs. generic MFMA is enforced by
# the predicate definitions themselves, not by branch order).

# Stage 1: classify the producer ONCE into an enum.
class ProducerKind(enum.Enum):
    PACK_MFMA = "pack_mfma"   # 4x4 PackMFMA (category=Pack*, rocisa=MFMAInstruction)
    MFMA      = "mfma"        # category=="MFMA"
    CVT_PACK  = "cvt_pack"    # category=Pack*, rocisa=VCvtPkF32toBF16
    ALU       = "alu"         # everything else _is_alu_producer accepts
    OTHER     = "other"       # falls through to wait-coverage

class ConsumerKind(enum.Enum):
    MFMA      = "mfma"        # _is_mfma_producer(c)
    CVT_PACK  = "cvt_pack"
    OTHER     = "other"

def classify_producer(p: GraphNode) -> ProducerKind: ...
def classify_consumer(c: GraphNode) -> ConsumerKind: ...

# Stage 2: dispatch table. One row per (producer, consumer) pair that
# carries a quad-cycle constraint. Missing key → fall-through to
# wait-coverage (or ALU early-return).
DISPATCH: Dict[Tuple[ProducerKind, ConsumerKind], Callable[..., TimingCheck]] = {
    (ProducerKind.PACK_MFMA, ConsumerKind.CVT_PACK): _mfma_pack_to_cvt_gap_ok,
    (ProducerKind.PACK_MFMA, ConsumerKind.MFMA):     _quad_cycle_gap_ok,
    (ProducerKind.PACK_MFMA, ConsumerKind.OTHER):    _quad_cycle_gap_ok,
    (ProducerKind.MFMA,      ConsumerKind.CVT_PACK): _quad_cycle_gap_ok,
    (ProducerKind.MFMA,      ConsumerKind.MFMA):     _quad_cycle_gap_ok,
    (ProducerKind.MFMA,      ConsumerKind.OTHER):    _quad_cycle_gap_ok,
    (ProducerKind.CVT_PACK,  ConsumerKind.MFMA):     _cvt_to_mfma_gap_ok,
    # CVT_PACK → CVT_PACK and CVT_PACK → OTHER fall through to ALU
    # exemption (no quad-cycle constraint).
    # ALU → * always falls through to ALU exemption.
}
```

The dispatch site:

```python
pk = classify_producer(p_node)
ck = classify_consumer(c_node)
helper = DISPATCH.get((pk, ck))
if helper is not None:
    check = helper(p_node, c_node, subj_graph)  # uniform signature
    if check.result == TimingResult.FAIL:
        return [TimingTooCloseFailure(...)]
    return []
if pk in (ProducerKind.ALU, ProducerKind.PACK_MFMA, ProducerKind.MFMA,
          ProducerKind.CVT_PACK):
    # ALU exemption (and all gap helpers that returned PASS above).
    return []
# Fall through to Phase-2 wait coverage.
```

### 3.2 Pros

- **Order-independent by construction.** The keys are pairwise disjoint
  (each edge maps to exactly one key, since the producer and consumer
  classifiers each return exactly one enum value). Adding a new pair
  is one row, not a re-derivation of where to insert a new branch.
- **De-duplicates `_classify_edge_coverage` and `diagnose_missing_edge`.**
  Both call sites can share the same dispatch helper. Saves ~60 LoC of
  duplicated branches.
- **Locally documented.** The dispatch table is one reference for
  "what gap applies to what edge"; the in-line "Must run BEFORE"
  comments become obsolete and can be deleted.
- **`_quad_cycle_gap_ok`'s `num_mfma_per_subiter` parameter** is unused
  (`CMSValidator.py:1753`); a refactor can drop it from the helper
  signature so all three gap helpers share the same
  `(producer, consumer, subj_graph) -> TimingCheck` shape, making the
  table value type uniform.

### 3.3 Cons

- **Two new classifier functions** (`classify_producer`,
  `classify_consumer`) with their own test surface. Each must
  faithfully reproduce the existing `_is_*_producer` semantics
  (PackMFMA carve-out from ALU, etc.) — these are the same edge cases
  the predicate chain was already encoding, so the audit is a
  refactor not a redesign.
- **`_quad_cycle_gap_ok` signature change.** `num_mfma_per_subiter`
  is currently a positional arg in test fixtures
  (`test_dataflow_graph_register_gaps.py:432`, `:514`, `:580`, etc.).
  Removing it requires updating ~10 call sites in tests. Per the bead's
  hard rule (no fallback shims), this is the right work, not extra
  work.
- **Producer/consumer enum exhaustiveness.** The `OTHER` catch-all on
  the consumer side is a bit ugly — most consumers aren't pre-classified
  by anything; the dispatch only cares about MFMA-vs-CVTPack. The
  consumer classifier is therefore "is this an MFMA, a CVTPack, or
  something else", which matches the existing predicate use but
  requires a new function purely to feed the table.

### 3.4 Order dependency: eliminated or relocated?

**Eliminated.** The keys are pairwise disjoint (each producer is
exactly one of the four enum values), so there is no order at all in
the dispatch step. The classifier functions themselves do have
ordering (e.g., `classify_producer` must check PackMFMA before MFMA
before CVT_PACK before ALU), but that ordering is local to the
classifier and is ENFORCED by the enum's exclusivity — not by which
branch runs first in a 50-line dispatch chain. A future refactor that
swaps the order inside `classify_producer` cannot silently regress
because the classifier returns an enum, and changing the enum
assignment for a producer either changes the dispatch result OR the
test assertion on what bucket that producer falls into. The fragility
moves from "branch order in dispatch" to "predicate order in
classifier" — and the latter is much smaller (3-4 lines), much more
testable (one classifier function, one assertion-per-bucket test),
and much easier to reason about (because each producer maps to
exactly one bucket, you can write
`assert classify_producer(make_packmfma()) is ProducerKind.PACK_MFMA`).

### 3.5 LoC estimate

- New module-level enums and classifier helpers: **~50 LoC**.
- New shared dispatch helper (`_dispatch_quad_cycle_check`): **~25 LoC**.
- Replace two dispatch chains in `_classify_edge_coverage` (lines
  `:2939–3012`, ~74 LoC) and `diagnose_missing_edge` (lines `:2660–2728`,
  ~70 LoC) with calls to the shared helper: net **−~120 LoC**.
- Drop unused `num_mfma_per_subiter` from `_quad_cycle_gap_ok`,
  update test fixtures: **~30 LoC** (mostly mechanical removal at
  test call sites).
- New unit tests for the classifier and dispatch table: **~80 LoC**.
- Delete the obsolete in-line "Must run BEFORE" carve-out comments
  in both call sites: **−~20 LoC**.

**Net: ~+45 LoC, ~−140 LoC = net −95 LoC**, plus the ~80 LoC of new
tests. Touching ~6 files (CMSValidator.py + 5 test files). Fits
"~50 LoC, no new conceptual surface" only if the tests don't count;
realistically this is a 200-line PR including tests. The bead's
auto-bundle threshold is "~50 LoC, no new conceptual surface, no test
fixture rewrites" — Option 1 fails the third condition (the
`_quad_cycle_gap_ok` signature change touches several test fixtures).

### 3.6 Runtime impact

Negligible. The current chain is 4 sequential predicate calls per
edge; the table is 2 classifier calls + 1 dict lookup per edge.
Classifier internals are the same predicate calls, just consolidated.
On the order of a few hundred nanoseconds per edge in either case.
`_classify_edge_coverage` runs once per edge in `compare_graphs`;
typical graphs have a few thousand edges per kernel; total runtime
contribution stays well under 1 ms.

### 3.7 Test impact

- `test_dataflow_graph_register_gaps.py` (89 refs): unaffected unless
  the `num_mfma_per_subiter` removal is bundled. If bundled, ~10 call
  sites updated mechanically (drop the positional arg).
- `test_arch_profile_unregistered_isa.py` (12 refs): unaffected.
- Other test files: unaffected.
- New tests (~80 LoC): one test per producer enum value (verify
  classifier maps the canonical producer correctly), one test per
  dispatch table entry (verify the right helper fires).

---

## 4. Option 2 — Producer/consumer kind enum + pattern match

### 4.1 Sketch

Same producer/consumer enum classification as Option 1, but instead of
a dict-of-tuples, use Python 3.10+ `match`/`case`:

```python
match (classify_producer(p_node), classify_consumer(c_node)):
    case (ProducerKind.PACK_MFMA, ConsumerKind.CVT_PACK):
        check = _mfma_pack_to_cvt_gap_ok(p_node, c_node, subj_graph)
        ...
    case (ProducerKind.MFMA, _) | (ProducerKind.PACK_MFMA, _):
        check = _quad_cycle_gap_ok(p_node, c_node, subj_graph)
        ...
    case (ProducerKind.CVT_PACK, ConsumerKind.MFMA):
        check = _cvt_to_mfma_gap_ok(p_node, c_node, subj_graph)
        ...
    case (ProducerKind.ALU, _) | (ProducerKind.CVT_PACK, _):
        return []  # ALU exemption / CVT-feeding-non-MFMA pass-through
    case _:
        pass  # fall through to Phase-2 wait coverage
```

### 4.2 Pros

- **Same order-independence properties as Option 1** (the cases are
  pairwise disjoint when written with literal tuples).
- **Reads as a decision table in source.** The `(pk, ck)` pairs sit
  next to their handlers in one syntactic block, no indirect dict
  lookup.
- **`_` wildcards capture the "everything else" cases naturally.**

### 4.3 Cons

- **Match cases ARE order-dependent in Python's `match` semantics.**
  Wildcard cases (`(ProducerKind.MFMA, _)`) match before more specific
  cases that come after them, so the writer still has to think about
  case order. The dispatch-table approach is strictly better here:
  the table key is the FULL pair, no wildcards needed.
- **Requires Python 3.10+.** The codebase already targets 3.10+
  (verified by checking `tensilelite/setup.py` if present), so this
  is not a hard blocker, but is a soft constraint worth confirming.
- **No de-duplication win for `diagnose_missing_edge`.** The match
  block is a syntactic structure; you can extract it to a helper, but
  then you've reinvented the dispatch table — one level of indirection
  on top of the match block, which makes Option 1 strictly simpler.
- **Same signature-uniformity work** as Option 1 (drop
  `num_mfma_per_subiter` from `_quad_cycle_gap_ok`).

### 4.4 Order dependency: eliminated or relocated?

**Relocated.** Match cases run top-to-bottom; the writer must order
specific cases before wildcard cases. This is the same fragility as
the current chain, expressed in slightly different syntax. Strict
elimination requires either (a) writing literal tuples for every
case (Option 1's table is the same thing in dict form), or (b) a
post-match validation step that asserts no case is shadowed.

### 4.5 LoC estimate

Roughly the same as Option 1 (~+50 / −120 / +80 tests). The
inability to extract the match block cleanly to a shared helper means
the duplicate dispatch in `diagnose_missing_edge` either stays
duplicated or gets factored differently from the
`_classify_edge_coverage` site. Net: marginally worse than Option 1
by ~30 LoC.

### 4.6 Recommendation: skip

Option 1 dominates Option 2 on every axis: same enum work, same
helper work, but a strictly-disjoint dict key replaces an
ordering-sensitive match, AND the dict can be passed by reference to
both call sites for de-duplication. No reason to prefer the match
form.

---

## 5. Option 3 — Visitor pattern

### 5.1 Sketch

Each producer kind owns its own `gap_check(consumer, graph)` method;
dispatch is by polymorphism on the producer-kind object.

```python
class ProducerKindHandler:
    def gap_check(self, c: GraphNode, g: DataflowGraph) -> TimingCheck:
        raise NotImplementedError

class PackMFMAHandler(ProducerKindHandler):
    def gap_check(self, c, g):
        if classify_consumer(c) is ConsumerKind.CVT_PACK:
            return _mfma_pack_to_cvt_gap_ok(self.producer, c, g)
        return _quad_cycle_gap_ok(self.producer, c, g)

class MFMAHandler(ProducerKindHandler):
    def gap_check(self, c, g):
        return _quad_cycle_gap_ok(self.producer, c, g)

class CVTPackHandler(ProducerKindHandler):
    def gap_check(self, c, g):
        if classify_consumer(c) is ConsumerKind.MFMA:
            return _cvt_to_mfma_gap_ok(self.producer, c, g)
        return TimingCheck.passing(0, 0)  # falls through to ALU

class ALUHandler(ProducerKindHandler):
    def gap_check(self, c, g):
        return TimingCheck.passing(0, 0)

# Dispatch:
HANDLER_FOR_KIND = {
    ProducerKind.PACK_MFMA: PackMFMAHandler,
    ProducerKind.MFMA:      MFMAHandler,
    ProducerKind.CVT_PACK:  CVTPackHandler,
    ProducerKind.ALU:       ALUHandler,
}

handler = HANDLER_FOR_KIND[classify_producer(p_node)](p_node)
check = handler.gap_check(c_node, subj_graph)
```

### 5.2 Pros

- **Per-kind locality.** Each producer kind owns its own gap check
  in one class. Adding a new kind = new class, new handler entry; no
  central dispatch to edit.
- **Naturally extensible.** If new producer kinds emerge (e.g., a
  hypothetical hardware change introduces a new visibility window),
  the visitor pattern absorbs them without touching any existing
  class.

### 5.3 Cons

- **Heavyweight for a 4-bucket dispatch.** Three new classes (or four
  with an ALU stub) for what's currently four `if` branches. The
  per-edge object allocation (one handler instance per
  `_classify_edge_coverage` call) adds GC pressure that the table
  approach doesn't.
- **The "dispatch order" problem reappears INSIDE each handler.**
  `PackMFMAHandler.gap_check` still has to choose between
  `_mfma_pack_to_cvt_gap_ok` and `_quad_cycle_gap_ok` based on
  consumer kind; that's a 1-line `if` rather than a 4-deep chain,
  but the conceptual problem ("dispatch by predicate, ordering
  matters") is still present in microcosm.
- **Doesn't fit the codebase's idioms.** The validator is otherwise
  function-oriented (`_is_*`, `_*_gap_ok`, `_classify_*`); introducing
  a one-off class hierarchy for dispatch is structural overhead the
  rest of the file doesn't pay.
- **Same signature-uniformity work** as Option 1.

### 5.4 Order dependency: eliminated or relocated?

**Relocated, partially eliminated.** The producer-kind dispatch is
order-independent (dict lookup on the enum). The
consumer-discrimination inside each handler is a 1-line `if`, which
is small enough to be a non-issue. Strict order-independence
requires extending the visitor to BOTH the producer and consumer
kinds — i.e., a double-dispatch (visitor of visitor), which is
overkill for a 4×3 grid.

### 5.5 LoC estimate

- New classes: **~80 LoC** (one per kind, plus a base).
- Handler dict: **~10 LoC**.
- Replace two dispatch chains: **−~140 LoC**.
- Same test impact as Option 1: **~+80 LoC**.

**Net: ~+90 LoC, ~−140 LoC = net −50 LoC**, +tests. Strictly worse
than Option 1 on LoC (visitor classes are bulkier than enum + table)
and worse on idiomatic fit.

---

## 6. Option 4 — Producer-classify-once + tiny dispatch (recommended path)

### 6.1 Sketch

Hybrid of Options 1 and 3, leaning on `rocm-libraries-009`'s direction.
Once the rocisa instruction object exposes a `category` enum attribute,
producer/consumer classification collapses to **one attribute read**.
The dispatch then becomes:

```python
# Post-009: category is one enum read, no predicate cascade.
def _quad_cycle_constraint_for(p: GraphNode, c: GraphNode) -> Optional[GapHelper]:
    """Return the gap helper for this producer/consumer pair, or None
    if the pair carries no quad-cycle constraint (ALU-immediate or
    Phase-2 wait coverage applies)."""
    pcat = p.semantic_category   # post-009: rocisa attribute
    ccat = c.semantic_category
    if pcat is SemanticCategory.PACK_MFMA and ccat is SemanticCategory.CVT_PACK:
        return _mfma_pack_to_cvt_gap_ok
    if pcat is SemanticCategory.PACK_MFMA or pcat is SemanticCategory.MFMA:
        return _quad_cycle_gap_ok
    if pcat is SemanticCategory.CVT_PACK and ccat is SemanticCategory.MFMA:
        return _cvt_to_mfma_gap_ok
    return None  # ALU-immediate / fall-through
```

This is structurally identical to Option 1's table but expressed as a
straight-line classifier function instead of a dict. The win is
specifically that `pcat` and `ccat` come from rocisa attributes
(post-009) so there's no predicate-chain layer to refactor.

### 6.2 Pros

- **Two enum reads per edge, three exact-match comparisons.** Order is
  trivially correct because each pair is matched by `is` against an
  enum singleton — no overlap.
- **Negligible LoC.** ~30 LoC for the function, replaces ~140 LoC
  across two call sites. Net ~−110 LoC.
- **Deletes `_is_mfma_producer`, `_is_cvt_pack_producer`,
  `_is_mfma_pack_producer`, and most of `_is_alu_producer` as part
  of `009`'s natural collapse.** This bead's work then becomes
  ~50 LoC of dispatch-site rewrite, not a 200-line refactor.
- **No new test fixtures.** Test fixtures already construct producers
  with the required category; once `009` lands they construct them
  with the new attribute too. Zero net test churn for THIS bead.

### 6.3 Cons

- **Depends on `009` landing first.** If `o0ei` is implemented before
  `009`, it has to either (a) re-derive `pcat`/`ccat` via the existing
  predicate cascade — collapsing to Option 1 in practice — or (b) wait.
- **`009` is in flight (status `in_progress`, P1)** but its dependency
  chain is closed (`8t9` and `br4` are both closed); it could land at
  any time. Order is uncertain.

### 6.4 LoC estimate

Conditional on `009` landing first:

- New classifier function: **~25 LoC**.
- Replace two dispatch chains: **−~140 LoC**.
- Test impact: **0 LoC** (009's tests already cover the category
  attribute; this bead's work is dispatch-site-only).

**Net: ~−115 LoC.** This is the cheapest option by a wide margin AND
fits the bead's auto-bundle threshold ("~50 LoC, no new conceptual
surface, no test fixture rewrites") if `009` lands first.

### 6.5 Order dependency: eliminated

Same argument as Option 1: pairwise-disjoint enum comparisons, no
overlap. Strictly stronger than Option 1 because the enum comes from
rocisa rather than being derived locally — there's no
`classify_producer` function to get wrong.

---

## 7. Interaction with bead 009

`rocm-libraries-009` ("Push semantic-category attribute onto rocisa
instruction classes") will:

- Add a `category` attribute to each rocisa instruction class returning
  a member of an `InstructionCategory` enum
  (LR/LW/GR/MFMA/SWAIT/SBARRIER/SNOP/SSETPRIO/PACK plus finer Pack
  subdivisions).
- Delete `_LR_CLASS_NAMES` ... `_CVT_PACK_CLASS_NAMES` (13 sets) and
  `_is_lr` ... `_is_cvt_pack` (10 discriminators) in
  `ScheduleCapture.py:969–1014`.
- Reduce `_is_mfma_producer`, `_is_cvt_pack_producer`,
  `_is_mfma_pack_producer`, and the bulk of `_is_alu_producer` to
  one-line attribute reads.

Effect on this bead:

- **The carve-out logic in `_is_alu_producer`** (the one that excludes
  PackMFMAs from the ALU bucket via `_is_mfma_pack_producer`,
  `CMSValidator.py:1917`) becomes a **single enum value distinction**
  in the rocisa attribute itself. The rocisa-side `category` attribute
  for a PackMFMA returns `PACK_MFMA` (or whatever the chosen finer
  Pack subdivision is) directly; the validator just reads it. The
  carve-out disappears.
- **The CVTPack carve-out** (the one in `_is_cvt_pack_producer`,
  `CMSValidator.py:1458–1482`) similarly disappears — `category`
  returns `CVT_PACK` directly.
- **Producer classification is a `getattr(p, "category")` call**
  rather than a 4-deep predicate cascade. No `classify_producer`
  function needed; the enum IS the classification.

This means the dispatch-table refactor (Option 1) is post-009 a
one-screen change: just the dispatch site itself, no new
classifier functions, no new tests. **It is materially cheaper to do
o0ei after 009 than before.**

If `009` lands FIRST: Option 4 (=Option 1 simplified) is ~50 LoC and
fits the bead's auto-bundle threshold; o0ei could be a single
follow-up commit.

If `o0ei` lands FIRST: Option 1 is ~200 LoC including tests and new
classifier functions; the work duplicates effort that `009` will then
delete (the new classifier helpers become one-line attribute reads
once `009` lands).

---

## 8. Recommendation

**File a follow-up bead** for the actual dispatch refactor. Do NOT
bundle implementation in the o0ei commit. Justification:

1. **`009` is in flight at P1.** Its predicate-collapse work
   materially changes which option is cheapest. Implementing o0ei now
   commits to Option 1 (the only option viable pre-009); waiting
   ~days/weeks lets us pick Option 4 (~−65 LoC vs Option 1, zero new
   conceptual surface).
2. **Option 1's auto-bundle threshold check fails.** The
   `_quad_cycle_gap_ok` signature change (drop `num_mfma_per_subiter`)
   touches multiple test fixtures
   (`test_dataflow_graph_register_gaps.py`'s 89-ref import surface,
   plus 4 other test files). The bead's bundle threshold is
   explicitly "no test fixture rewrites." Option 1 is too big to
   bundle into an investigation commit.
3. **The audit memo itself fixes the immediate fragility risk.** The
   in-line "Must run BEFORE" comments at `:2942–2947` and `:2987–2996`
   are correct and load-bearing. With this audit checked in, future
   refactors have a written reference for WHY the order is fragile and
   what the recommended replacement is. The actual refactor can wait
   for `009` without losing institutional memory.

### 8.1 Follow-up bead spec

Title: "Implement Option 4: replace order-dependent dispatch in
`_classify_edge_coverage` and `diagnose_missing_edge` with
category-based table"

Body: see §6 of this memo. Acceptance:

- Delete the 4-branch dispatch chain at `CMSValidator.py:2939–3012`
  (`_classify_edge_coverage`).
- Delete the 4-branch dispatch chain at `CMSValidator.py:2660–2728`
  (`diagnose_missing_edge`).
- Replace both with one shared helper that consults a single dispatch
  table or function keyed on (producer.category, consumer.category).
- Drop the obsolete in-line "Must run BEFORE" carve-out comments at
  both call sites.
- Drop `num_mfma_per_subiter` from `_quad_cycle_gap_ok` (now-uniform
  3-arg signature: `(producer, consumer, subj_graph) -> TimingCheck`).
  Update test fixtures.
- All 677/2/1 baseline tests still pass.
- Pytest count delta should be at most ±2 (one or two new
  classifier-table tests).

Dependencies: `rocm-libraries-009` (`blocks`).

Effort estimate: **~50 LoC + ~30 LoC test churn = 80 LoC**, single
PR, ~2 hours of focused work after `009` lands.

### 8.2 If `009` slips for >1 month

Re-evaluate. If `009` is going to slip past, say, July 2026, then
Option 1 (without the rocisa attribute) is worth implementing as a
holding pattern. The two refactors compose cleanly: a Option 1
dispatch table written today gets simplified by `009`'s arrival
(the classifier helpers become one-line attribute reads), but the
dispatch site itself is unchanged. So a "Option 1 now, Option 4 cleanup
when `009` lands" sequence is also viable if `009` slips.

### 8.3 What this commit DOES land

- This memo (`Tensile/Components/QUAD_CYCLE_DISPATCH_AUDIT.md`).
- A commented update to `_classify_edge_coverage` and
  `diagnose_missing_edge` that adds a TODO pointing at the follow-up
  bead and this memo, so the next refactorer sees the written
  recommendation in-line. (Optional — see §9.)

No code changes to the dispatch logic itself. Investigation only.

---

## 9. Open questions for the user

1. **Bundle a TODO comment update?** The two dispatch sites have
   "Must run BEFORE" comments documenting the fragility but no
   pointer to this audit or the (forthcoming) follow-up bead. Worth
   landing a 4-line comment update in this commit pointing future
   refactors at the audit and at the follow-up bead ID, OR is the
   investigation commit's own existence + bead linkage sufficient
   trail? (Default if no answer: do NOT bundle the comment update —
   keep this commit memo-only.)
2. **Confirm the recommendation to defer to post-009.** This
   recommendation rests on `009` being a near-term landing; if `009`
   is going to slip past August 2026, Option 1 today is preferable
   to waiting indefinitely. The user has visibility into `009`'s
   schedule that the agent does not.
3. **Should the follow-up bead block on `009` strictly, or merely
   recommend post-009 sequencing?** A strict block (`dep add o0ei-fu
   009 --type blocks`) means the follow-up cannot run until `009`
   closes. A soft sequencing recommendation (in the bead body) lets
   the follow-up agent pick Option 1 if `009` is delayed without a
   second consultation. Default if no answer: soft recommendation.
