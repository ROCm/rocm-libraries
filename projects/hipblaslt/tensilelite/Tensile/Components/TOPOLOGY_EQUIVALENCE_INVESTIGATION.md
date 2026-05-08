# Topology-equivalent comparison in `compare_graphs` — design investigation

Bead: `rocm-libraries-wx9.3` (sub-tasks 7a + 7b).

This memo enumerates concrete approaches for replacing today's
register-name-based `compare_graphs` with a topology-equivalent
comparison. **Phase 1 — research only. No code changes.**

---

## 0. Today's situation in one screen

`Tensile/Components/CMSValidator.py`:

- `GraphNode.identity = (class_tag, loop_index, _canonical_render(inst))`
  — third element is the rendered assembly string of the instruction,
  which contains register-name text. (`_identity_for`, l.772-802;
  `_canonical_render` lives in `ScheduleCapture.py`, l.980-1010.)
- `DataflowEdge` carries `(producer, consumer, resource, edge_kind)`
  where `resource` is either a rocisa `RegisterContainer` (numeric or
  symbolic) or a `MemoryRegion` (frozen dataclass).
- `DataflowGraph.edge_keys()` returns
  `{(p.identity, c.identity, e.resource, e.edge_kind) for e in edges}`
  (l.471-474).
- `compare_graphs(reference, subject)` (l.2338-2414):
    1. Bails early with `CaptureConsistencyError` if the data-flow node
       identity sets differ.
    2. `missing = ref.edge_keys() - subj.edge_keys()`.
    3. Routes each missing key through `diagnose_missing_edge`, which
       walks Phase 1 (`OrderInvertedFailure`), an SCC clobber check
       (`OverriddenInputFailure`), MFMA / CVT / ALU exemption branches,
       and Phase 2 wait/barrier coverage. An unclassified miss raises
       `UnexplainedMissingEdgeError`.

`ScheduleCapture.py`:

- `_canonical_render(inst)` is `str(inst)` minus comments/whitespace —
  contains symbolic *and* numeric register names verbatim.
- `_byte_keys_for_resource(resource)` already produces a
  topology-style key per byte: numeric vgprs key as `(regType, idx+i)`,
  symbolic as `(regType, name, base+i)`, memory as
  `("mem", space, buffer_id, offset+i)`. **Notably, intra-graph,
  numeric and symbolic forms of the same register live in *different
  byte-key namespaces today.* That is why the latest-writer resolver
  cannot wire a symbolic-write to a numeric-read within one graph
  (the 7a problem).**

**Failure classes** (l.1971-2232): `OrderInvertedFailure`,
`MissingWaitFailure`, `WaitInsufficientFailure`,
`MissingBarrierFailure`, `TimingTooCloseFailure`,
`InvalidCounterValueFailure`, `OverriddenInputFailure`. All carry
`FailureNodeLabel`s — `cms_node_label` produces strings like
`'PackA0[2]'` from the *subject* node's body capture, not from
register names. So **the diagnostic surface is already
register-name-free**; only the *matching* surface (identity tuples,
edge keys) leaks register names.

**Key invariant.** Per the bead's wx9.3 reframe (2026-05-08): in
production, the two graphs SHOULD be identical 99% of the time.
Differences are rare and indicate a real CMS-scheduling bug or a
validator bug. Designs with a fast-path return on match and an
expensive-but-precise mismatch-explanation path are strongly preferred.

---

## 1. Catalog of approaches

For each approach: **mechanism**, **fast-path cost**,
**mismatch-path cost**, **failure granularity**, **determinism**,
**implementation complexity**, **fit with the 99%-identical
constraint**, **intra-graph (7a) implication**.

---

### Approach A — Multiset of structural edge signatures

**Mechanism.** For every edge, compute
`sig = (src_role, src_position, sink_role, sink_position, edge_kind, byte_offset_tuple)`
where `src_role` / `sink_role` are derived from the node's category
(`PackA0`, `LRA1`, `MFMA`, …) and `*_position` is the
`SchedulePosition` (loop_index, vmfma_index, sub_index) — a stable
key that survives both register renaming and the numeric/symbolic
form distinction. Compare the two multisets. Mismatch =
symmetric difference.

**Fast path** (graphs identical): build both multisets, compare —
O(E) work, **always touches every edge**. There is no free
short-circuit.

**Mismatch path:** symmetric difference yields the exact differing
signatures with full role/position/byte detail.

**Failure granularity:** very fine. "Edge from PackA1@vmfma=14 to
MFMA@vmfma=18 with byte_offset=0..15 missing in subject" maps
1:1 to today's `Failure` triggers.

**Determinism:** stable (multiset equality is order-free; sets sort
deterministically when needed for diagnostics).

**Implementation complexity:** ~150 LoC in `compare_graphs` plus a
`_structural_signature(edge)` helper. Replaces `edge_keys` and the
`ref_keys / subj_keys` diff. `diagnose_missing_edge` keeps its
classifier dispatch verbatim — it only loses its
`identity`-keyed lookup of the subject node, which becomes
`(role, position)`-keyed instead.

**99% fit:** weak — pays full O(E) every time. No hash short-circuit.

**Intra-graph (7a) implication:** the graph builder must canonicalize
operand references so that one `(regType, name|idx, base, count)`
identity is reached regardless of whether the rocisa container is
symbolic or numeric. Otherwise A's byte_offset_tuple namespaces
differ across forms, just as today's `_byte_keys_for_resource`
does. **A does not on its own solve 7a — it requires a builder fix.**

---

### Approach B — Per-graph hash + descend-on-mismatch (Merkle-style or sorted-edge digest)

**Mechanism.**
- Compute a 128-bit BLAKE2b digest over the sorted multiset of
  structural edge signatures from Approach A. Cache on the
  `DataflowGraph` (`_topology_digest`).
- `compare_graphs(ref, subj)`: if `ref.digest() == subj.digest()`,
  return `[]` immediately.
- On mismatch, fall into Approach A's set-diff and the existing
  `diagnose_missing_edge` pipeline.

**Fast path:** O(E) the **first** time each digest is computed, then
**O(1)** on subsequent comparisons. With caching at graph build, the
hot path of "compare_graphs called on a freshly built pair" is one
hash equality test plus the cost of building each digest once.

**Mismatch path:** identical to Approach A.

**Failure granularity:** identical to Approach A on mismatch.

**Determinism:** stable iff (a) the signature multiset is built from
sortable keys (it is) and (b) the hash is reproducible (BLAKE2b is).
Critically — `hash()` builtins must NOT be used; PYTHONHASHSEED
randomizes them across runs.

**Implementation complexity:** ~200 LoC (Approach A + a digest
helper + cache plumbing on `DataflowGraph`).

**99% fit:** strong. Single 128-bit equality check skips the entire
diff/diagnosis pipeline on the common path. The slow-path is
factored as a separate method called only when the cheap test
disagrees.

**Quantitative figures (rough, pre-benchmark estimates).** At a
typical graph size of E≈500 edges:

- **BLAKE2b digest** over a serialized sorted edge tuple of 500
  elements: ~5-15µs single-shot, then **0** for every cached read
  (a Python attribute access).
- **Set difference** of 500 4-tuples (Approach A's hot path):
  ~50-100µs per call (Python set hashing + element walk).
- **Tuple comparison** of 500 4-tuples (Approach J's hot path):
  ~50-100µs per call when equal (full element walk in C);
  short-circuits faster on mismatch.
- **Per-comparison cost amortized over N calls against the same
  pair of graphs:**
  - A: `N * 50-100µs ≈ 75N µs`.
  - J: `N * 50-100µs ≈ 75N µs`.
  - B: `2 * 10µs (digest build, once per graph) + N * <0.1µs ≈
    20µs + 0.1N µs`.

  B breaks even with A and J at N≈1, then dominates linearly. The
  validator pipeline calls `compare_graphs` multiple times across
  passes for the same `(reference, subject)` pair, so N is
  realistically ≥1 in production. **This is the empirical
  justification for "hash + cache" over "structural compare per
  call."**

**Intra-graph (7a) implication:** same as A — canonicalize at the
builder. The digest is over signatures, so any form-distinction
the builder leaks is preserved in the digest.

---

### Approach C — Canonical-renaming + verbatim compare (α-conversion)

**Mechanism.** Walk each graph in topological emit order
(SchedulePosition lex-sort). Maintain an integer counter; the first
time a register is *defined* by some node, assign it
canonical-id `r{counter++}`. Every subsequent reference to that
register (read or write) uses the canonical-id. After this pass,
every edge's `resource` is replaced with its canonical-id.
Compare the two graphs verbatim with today's `edge_keys()` machinery.

**Fast path:** O(V+E) renaming pass on each graph, then a set diff.
No free short-circuit (still touches every edge).

**Mismatch path:** standard set-diff against the renamed edges.

**Failure granularity:** fine, but the `resource` field on the
`Failure`-bound edge becomes the canonical-id (`r17`), which is
*less* informative than today's `vgprValuA_X0_I0` for human
debugging. We would want to keep both — original for display,
canonical for matching.

**Determinism:** stable iff the topological walk is total-ordered
(it is — `SchedulePosition` lex-sorts by `(loop_index, vmfma_index,
sub_index)`).

**Implementation complexity:** ~300 LoC. A new `canonicalize_dataflow`
pass on `DataflowGraph`, plus dual storage of original-and-canonical
resources on `DataflowEdge`. Touches the rule registry's edge
emission path.

**99% fit:** weak — pays O(E) renaming each time.

**Relationship to B (corrected framing).** C is **not** a competitor
to B; it is the canonicalization step B's hash digest sits on top of.
Concretely:

- **B = (canonicalize) + (hash + cache).** The hash is computed
  over canonicalized signatures; canonicalization is required to
  reach a form whose hash is meaningful.
- **C = (canonicalize) + (verbatim compare).** Same canonicalization
  step, no hash layer.

What B adds over C is the hash-and-cache layer that turns matched
comparisons into O(1). C's only fast-path option without a hash is
Python tuple short-circuit on equality — which is exactly Approach J.
So practically, C without B's hash layer reduces to J. Picking B
means picking the canonicalization (which C contributes) **and** the
hash-cache layer on top.

**Intra-graph (7a) implication:** C's renaming pass implicitly fixes
7a — the first-write-wins canonical-id assignment maps both symbolic
and numeric references for the same physical register to the same
counter slot, *provided* the builder collapses them into one byte_key
namespace before the rename runs. Without that, two distinct
"definitions" still emit, and 7a returns. So C still requires the
same builder fix as A and B.

---

### Approach D — Bisimulation

**Mechanism.** Two graphs are equivalent iff there exists a relation
`R` on nodes such that R-related nodes have R-related successors
under matching edge labels (kind + byte_offset; **not** register
name). Compute the coarsest such `R` via Paige–Tarjan partition
refinement. Two graphs equivalent iff the initial nodes (sources:
kernel-arg loads; sinks: stores) are R-related and the partition
covers all of both graphs.

**Fast path:** no free short-circuit. Partition refinement is
O((V+E) log V) every time.

**Mismatch path:** the partition-refinement final state tells you
*which* equivalence classes split — you can identify "the consumer
node at vmfma=18 is in a class of size 1 in subj but size 2 in ref".
That is structural diagnostic information of a different shape than
today's per-edge `Failure` classes.

**Failure granularity:** classes-and-cells, not edges. The current
`OrderInvertedFailure` / `MissingWaitFailure` shape doesn't fall out
naturally; you would have to translate refinement-cell deltas into
edge-level diagnostics, which is awkward.

**Determinism:** stable (partition refinement is a fixpoint).

**Implementation complexity:** ~500-800 LoC plus a full rewrite of
`diagnose_missing_edge` to consume cell deltas. The diagnostic
translation is the expensive part.

**99% fit:** weak — no fast path; same cost on match as on mismatch.

**Intra-graph (7a) implication:** bisimulation respects whatever the
builder gives it. If the builder leaves symbolic and numeric
references in different namespaces, bisimulation still fails to
identify them as the same node. So D requires the same builder fix.

**Verdict.** D is strictly dominated by B+A on every axis for our use
case. The two graphs are stipulated to have identical topology in
the success case; bisimulation pays a refinement loop's worth of
overhead to discover what a hash equality test resolves in O(1).
**Exclude from the recommendation.**

---

### Approach E — Graph isomorphism (VF2 / nauty / bliss)

**Mechanism.** Run a full graph isomorphism algorithm
(VF2 in NetworkX, or pynauty/pybliss bindings). Returns a vertex
permutation if the graphs are isomorphic; returns None / raises
otherwise.

**Fast path:** none — VF2 is exponential worst case; nauty/bliss are
super-polynomial worst case but practically fast on
asymmetric/coloured graphs. Always touches the whole graph.

**Mismatch path:** "not isomorphic" with no localization. Diagnostic
output requires post-hoc structural diffing on top.

**Failure granularity:** terrible — bare yes/no. To localize, you
have to fall back to A or F anyway.

**Determinism:** stable but library-dependent.

**Implementation complexity:** medium for VF2 (NetworkX is already a
heavyweight dependency to pull in, currently not in `pyproject.toml`).
High for nauty/bliss (C library + Python binding).

**99% fit:** weak — no fast path, and the slow path is the *most*
expensive of any approach considered.

**Verdict.** Pure overkill: our graphs are stipulated to have
identical topology in the common case AND identical node labels (via
`(role, position)`); we do not need a permutation search.
**Exclude from the recommendation.**

---

### Approach F — Topological digest tree (Merkle hash per node)

**Mechanism.** Hash each node from `(role, label, sorted hashes of
incoming edges with their (kind, byte_offset, source-hash))` —
processed in topological order so source-hashes are available when a
sink is hashed. Compare the per-node hash multiset. Mismatch
locality = which subtree's hashes diverge.

**Fast path:** O(V+E) topological walk on each graph; final
multiset hash comparison is then O(1). With caching, this is the
same fast-path cost as B in steady state — the difference is that
F's per-node hashes give finer mismatch localization than B's
single graph digest.

**Mismatch path:** binary-search descent through differing per-node
hashes localizes the first divergent node in O(V) time and gives the
exact predecessor whose hash flipped.

**Failure granularity:** very fine. Translates naturally into the
existing `OrderInvertedFailure` / `OverriddenInputFailure` shape
because per-node-hash divergence pinpoints producer/consumer pairs.

**Determinism:** stable with BLAKE2b.

**Implementation complexity:** ~250 LoC. Topological hashing pass +
mismatch-descent helper.

**99% fit:** strong, **same** fast-path cost as Approach B (one hash
compare). The difference is mismatch-path quality: B falls back to
A's edge-set diff; F localizes through the per-node hash tree.

**Intra-graph (7a) implication:** F's per-node hash inputs include
the rocisa container reference; if the *container instance* used
for the symbolic and numeric form of one logical register is the
same Python object (which it isn't today — they're two distinct
constructor calls), F could side-step the builder fix. In our world
they're always distinct objects, so F still requires the builder
to collapse them into one byte-key namespace.

**Edge case.** Computing a per-node hash that incorporates incoming
edges requires a deterministic ordering on those edges. We sort by
`(kind, byte_offset_tuple, source_hash)` — a total order.

---

### Approach G — Iceberg / two-tier (cheap fingerprint then detailed compare)

**Mechanism.** Compute a cheap fingerprint per graph: a tuple like
`(num_nodes_per_role, num_edges_per_kind,
total_byte_coverage_per_kind)`. Compare fingerprints first; on
match, run a more detailed compare; on mismatch, short-circuit to a
high-level diff ("subj has 5 fewer LRA1→MFMA edges than ref").

**Fast path:** O(V+E) fingerprint each time (cheaper constant
factor than A or F's signatures), **no hash equality short-circuit
unless we ALSO cache** — at which point G converges to B.

**Mismatch path:** fingerprint difference yields a coarse summary;
need to descend through A or F for per-edge attribution.

**Failure granularity:** coarse at the fingerprint level, fine after
descent.

**Determinism:** stable.

**Implementation complexity:** ~120 LoC for the fingerprint plus
whatever the second-tier compare is (so essentially A or F bolted on
top). Net: more code than B, less precise than B+F.

**99% fit:** medium. G is what you build if you can't compute or
cache a full digest — it's the "cheap structural check first"
pattern with weaker discrimination than a hash. Once you've decided
to cache `_topology_digest` on the graph (B), G's fingerprint is
strictly less precise (different graphs can have the same
fingerprint).

**Verdict.** Strictly dominated by B as a fast path, and dominated
by F as a localized-mismatch tool. The fingerprint *could* be useful
as a single-call summary when the human reading a diff wants
"here's the gross size of the divergence" before scrolling through
edge-level detail. Keep as a candidate diagnostic helper, not as the
primary comparison mechanism.

---

### Approach H — Dictionary-of-edges keyed by `(src_role, sink_role, kind)`, values = ordered byte-offset lists

**Mechanism.** For each graph, group edges into
`{(src_role, sink_role, kind) -> sorted list of byte-offset tuples}`.
Compare the dicts. Differences localize to specific role-pairs:
"more LRA0→MFMA edges in ref than subj at byte_offset=0..3".

**Fast path:** O(V+E) dict construction each time. Dict equality
(`==`) on the constructed groups in Python is O(E) amortized but
short-circuits on first mismatch — *some* practical speedup but
proportional to the size of the first differing bucket, not the
graph as a whole.

**Mismatch path:** per-bucket diff is naturally human-readable
("this role-pair has N edges in ref vs M in subj").

**Failure granularity:** medium — coarser than per-edge but better
grouped than per-graph.

**Determinism:** stable.

**Implementation complexity:** ~180 LoC.

**99% fit:** weak — no hash short-circuit. Equivalent to A on the
fast path.

**Verdict.** H's value is **diagnostic grouping**, not raw
comparison. **Folded into F's output formatting** (see §6
recommendation): F's descent yields per-node divergence; the
output formatter groups those divergences by `(src_role, sink_role,
kind)` exactly as H prescribes ("subj is missing 12 edges, all
sharing role-pair LRA1→MFMA, byte offsets 0,4,8,..."). H is no
longer a separate "approach" — it is the presentation layer of F's
diagnostic output.

---

### Approach I — Bloom-filter early-out (briefly considered)

**Mechanism.** Compute a Bloom filter of edge signatures for one
graph; query the other graph's signatures against it. False-positive
rate tunable.

**99% fit:** medium. False-positive risk forces a verification pass
on success, defeating the fast-path advantage.

**Verdict.** Strictly dominated by B (a single 128-bit hash compare
has no false positives for our small E ≈ 10²-10³ edges).
**Exclude.**

---

### Approach J — Sorted canonical tuple, direct compare (no hash)

**Mechanism.** Same as A but the edges are stored in a **sorted
tuple** (not a multiset/set). Comparison is `tuple_a == tuple_b`,
which short-circuits on first inequality but walks the entire
tuple when equal.

**Fast path (graphs equal):** O(E) per call. Python's `tuple ==
tuple` is implemented in C and walks element-by-element, so on
matched graphs it pays full E comparisons every call.

**Mismatch path:** short-circuits on the first differing element;
diagnostic output is "first differing edge index", which gives less
context than A's full symmetric difference (you only see one
mismatch, not the whole set).

**Failure granularity:** per-edge but only the first divergent edge.

**Determinism:** stable.

**Implementation complexity:** ~120 LoC. Less than A because no
multiset machinery.

**99% fit:** weak — pays O(E) per call on matched graphs; no
caching.

**Versus B (the actual quantitative justification).** At E≈500,
sorted-tuple comparison costs roughly 50-100µs per call (Python
C-level tuple equality, no Python attribute access on hashable
elements). BLAKE2b over the same sorted serialization costs roughly
5-15µs **once** during digest computation, then O(1) on every
subsequent comparison. Across N comparator invocations against the
same pair of graphs (which happens routinely as the validator
iterates), the cumulative cost is:

- J: `N * 50-100µs ≈ 50N µs`.
- B: `2 * 10µs (one per graph, build-time) + N * <0.1µs ≈ 20µs + 0.1N µs`.

J wins only at N=0 (no comparison at all). At N=1 the costs are
comparable. At N≥2 B dominates linearly. Since the validator
pipeline runs the same `(reference, subject)` pair through multiple
classifier passes per `compare_graphs` call (and `compare_graphs`
is itself called multiple times across the validator suite), N is
in practice ≥1 and growing. **B's hash-and-cache wins because the
cost amortizes; J's sort-and-compare does not.**

---

### Approach K — Parallel emit-stream walk

**Mechanism.** Both captures come from the **same kernel-writer
state in the same build**. Beyond topological equivalence, they
are step-by-step parallel emissions of the same writer. A
comparator can simply zip the two emit streams and compare
instruction-by-instruction; the first mismatched step is the
divergence point.

**Fast path:** O(N) where N is the number of emitted instructions;
short-circuits on first mismatch.

**Mismatch path:** "step k of body B differed: ref emitted X, subj
emitted Y" — natural localization with full per-step context.

**Failure granularity:** per-emit-step (effectively per-instruction).
Maps to producer/consumer pairs trivially because each step is
exactly one instruction with its operands.

**Determinism:** stable iff emission order is total-ordered (it
is — `SchedulePosition` lex-sort, same as elsewhere in the memo).

**Implementation complexity:** ~80 LoC if streams are still
available at comparison time. Effectively zero new analysis
machinery.

**99% fit:** strong on match (single linear walk; no hash setup),
**stronger than B+F on mismatch** (gives per-step localization for
free without a separate descent helper).

**Availability — the open question.** Are the captures still held
as ordered emit streams at comparison time? Reading the code:
`FourPartCapture` exposes captured bodies, but by the time
`DataflowGraph` is built (`build_dataflow_graph` in
`CMSValidator.py`) the streams have been consumed to construct
nodes and edges. **If the captures are no longer ordered streams
at `compare_graphs` time, K is unavailable.** If they are still
addressable in emission order (e.g., on the underlying
`FourPartCapture`), K may dominate B+F by being simpler, requiring
no digest plumbing, and giving better mismatch localization.

**Verdict.** This depends on whether `compare_graphs` can reach
back to the ordered emit stream. If yes, K is a serious contender.
If no, K is unavailable and B+F is the right path. **The user
should resolve this before B+F implementation lands** — it's
cheaper to discover K is viable now than to build B+F and then
realize a simpler comparator was always available.

---

### Summary table

| Approach | Fast-path (match) | Mismatch path | Granularity | Det. | LoC | 99% fit |
|---|---|---|---|---|---|---|
| A — multiset of edge signatures | O(E) | O(E) set diff | per-edge | yes | ~150 | weak |
| **B — graph digest + descend** | **O(1) cached** | A's diff | per-edge | yes | ~200 | **strong** |
| C — canonical-rename + diff | O(V+E) | O(E) | per-edge | yes | ~300 | weak |
| D — bisimulation | O((V+E)log V) | classes | cells | yes | ~600 | weak (excluded) |
| E — graph iso (VF2/nauty) | exponential | none | yes/no | yes | ~400 + dep | weak (excluded) |
| **F — Merkle per-node tree** | **O(1) cached** | O(V) descent | **per-node** | yes | ~250 | **strong** |
| G — iceberg fingerprint | O(E) | A/F descent | medium | yes | ~120 | medium |
| H — dict-of-edges by role-pair | O(E) | per-bucket | grouped | yes | ~180 | weak |
| I — Bloom early-out | O(E) | A's diff | per-edge | yes | ~120 | medium (excluded) |
| J — sorted tuple, direct compare | O(E) per call | first diff only | per-edge | yes | ~120 | weak |
| K — parallel emit-stream walk | O(N), short-circuits | first diff step | per-step | yes | ~80 | strong if available |

---

## 2. The 99%-identical property — explicit treatment

For each viable approach (A, B, C, F, G, H):

| Approach | True fast-path on match? | Fast-path cost | Slow-path factorable? |
|---|---|---|---|
| A | no | O(E) every time | n/a (no slow path layer) |
| **B** | **yes — single 128-bit compare** | **O(1) after cache populated** | **yes — slow path is A** |
| C | no | O(V+E) rename + O(E) diff | no — coupling renamer to diagnostic shape |
| **F** | **yes — single hash compare** | **O(1) after cache populated** | **yes — descent is a separate helper** |
| G | partial (fingerprint compare is O(roles)) | O(E) build + O(roles) compare | yes — A or F as the descent |
| H | no | O(E) every time | n/a |

**Only B and F have an O(1) fast path on graphs that compare equal**,
*provided* the digest is computed once at graph-build time (or
lazily on first request and memoised). Both require the digest to be
deterministic across runs (no `hash(str)` — use BLAKE2b).

**B and F can be combined**: cache a top-level digest (B) AND the
per-node hash table (F) on the graph. Top-level compare is the
fast path; on mismatch, walk the per-node hashes for localization.

---

## 3. Diagnostic quality — what each approach can say when graphs differ

Today's diagnostic surface emits typed `Failure` instances:
`OrderInvertedFailure`, `MissingWaitFailure`,
`WaitInsufficientFailure`, `MissingBarrierFailure`,
`OverriddenInputFailure`, `TimingTooCloseFailure`,
`InvalidCounterValueFailure`. Each carries `FailureNodeLabel`
producer/consumer pairs **rendered from the subject's body capture**
(via `cms_node_label` — l.2251) — *not* from register-name strings.
That means the diagnostic surface is already register-name-free; the
typed-`Failure` shape can survive any of the topology approaches
without changes, provided the matching layer can locate the
producer/consumer nodes by role+position.

| Approach | Minimal differing element | Group differences | Survives current `Failure` hierarchy? |
|---|---|---|---|
| A | one edge | by post-processing into role-pairs (see H) | yes — `diagnose_missing_edge` keeps its dispatch |
| B | one edge (via A on slow path) | same | yes |
| C | one edge | same | yes — but `Failure.resource` becomes a canonical-id, less debuggable |
| D | partition cell | naturally | no — cell-deltas don't map to today's `Failure` shape |
| E | none | none | no — bare yes/no |
| F | one node + first divergent predecessor | naturally (per-node hash buckets) | yes — descent yields producer+consumer pairs |
| G | role-pair counts | naturally | yes (after F descent) |
| H | role-pair group | natively | yes (after per-bucket descent) |

**B+F** combined gives the best diagnostic chain:
1. B — cheap match check.
2. F — on mismatch, localize to the first divergent producer/consumer
   and emit the divergence grouped by `(src_role, sink_role, kind)`
   (the role-pair grouping formerly tracked as Approach H, now
   absorbed into F's output formatting).

The existing `diagnose_missing_edge` classifier keeps its phased
dispatch (Phase 1 OrderInverted, SCC clobber check, MFMA/CVT/ALU
exemption, Phase 2 wait/barrier coverage). It needs ONE input change:
instead of looking the subject node up by `identity`, it looks it up
by `(role, position)`. Everything downstream — `cms_node_label`,
`_cms_iter_delta`, the wait/barrier helpers — already runs off
`(category, position)`, not register names.

---

## 3a. Coverage notes — handling pieces the recommendation glosses over

### Missing-node handling

The B+F recommendation handles edge-multiset differences via F's
descent. It does not directly address what happens when subject has
a node that is missing entirely from reference (or vice-versa). Two
sub-questions:

1. *Where does the existing `CaptureConsistencyError` pre-check
   live?* (Today, around `CMSValidator.py:2371`, inside
   `compare_graphs`.) Three options:
   - Keep it inside `compare_graphs`, as a guard that runs before
     the digest equality test. Cheap (one set difference on node
     identity sets) and preserves today's "validator-bug vs
     CMS-scheduling-bug" distinction.
   - Fold the node-identity multiset into the digest input. A
     missing-node difference would then surface as a digest
     mismatch indistinguishable from a missing-edge difference;
     localization would still work (F's per-node hashes would diverge
     at the missing node) but the dedicated `CaptureConsistencyError`
     framing is lost.
   - Move it to a separate validator pass that runs upstream of
     `compare_graphs`. Cleanest separation but adds a pass.

   **Recommendation (memo author's tentative):** keep the pre-check
   inside `compare_graphs` as a guard. The check is O(V) and
   discriminates capture-pipeline bugs from real CMS-scheduling
   bugs — that distinction is worth preserving.

2. *How does F's descent handle nodes that exist on one side but not
   the other?* The asymmetry is real: a missing node has no per-node
   hash, so the descent cannot match it against anything. The
   intended contract is that F's descent is invoked **only after**
   the node-identity pre-check passes — i.e., both graphs are
   guaranteed to have the same node identity multiset. Differences
   F sees are therefore guaranteed to be edge-only differences
   reachable from a node that exists on both sides. The pre-check is
   a precondition for F's descent.

### FourPart per-codepath dimension

The validator's `DataflowGraph` is built from a `FourPartCapture`
with four bodies (`main_loop`, `main_loop_prev`, `n_gl`, `n_ll`).
The digest scheme must handle this:

- *One digest covering all four bodies, or one digest per body?*
  **Recommendation: one digest per body, plus an outer digest over
  the four sub-digests.** Per-body digests give F's descent a free
  first localization step ("the difference is in `n_gl` only");
  the outer digest preserves the O(1) fast-path when all four bodies
  match.
- *If two graphs differ only in the `n_gl` body, can F's descent
  localize?* Yes, with the per-body scheme above: the outer
  digest mismatches, the per-body digests show three matches and
  one mismatch (`n_gl`), and F's per-node descent runs only inside
  `n_gl`. Without the per-body sub-digests, F still reaches the
  same answer but pays a walk over all four bodies' nodes to find
  the divergence.

### `_classify_edge_coverage` interaction

`_classify_edge_coverage` (`CMSValidator.py:486, 733, 1391, 1406,
1416, 2475, 2781, 2789`) operates within a single graph: given an
edge, it classifies coverage relative to that edge's own
producer/consumer context (waits, barriers, exemptions). It does
not consume cross-graph identity tuples — its inputs are an edge
and the graph it lives in. It is therefore **independent of the
7b reframe** and untouched by the B+F migration; it continues to
run on whichever graph holds the edge being classified.

---

## 4. Intra-graph register identity — the 7a problem

The pinned test `test_symbolic_and_numeric_for_same_logical_reg_unchanged`
(`test_dataflow_graph_comparison.py:499-516`) demonstrates: within
one graph, `vgpr("ValuA_X0_I0", 4)` and `vgpr(8, 4)` for the same
physical register today produce different `_canonical_render`
strings → different identity tuples → the latest-writer resolver in
`build_dataflow_graph` Phase 2 wires them as separate writes/reads
even when one instruction's symbolic vgpr is the same physical
register as another's numeric vgpr.

**This is a graph-builder problem, not a comparator problem.** The
edges that *should* exist between a symbolic-write and a numeric-read
of the same register simply aren't built today. No comparator — A
through H — can match an edge that doesn't exist on either side.

For each cross-graph approach, the intra-graph requirement is:

| Approach | Intra-graph 7a fix required? | Why |
|---|---|---|
| A | yes — collapse byte_keys at builder | Otherwise byte_offset namespaces still differ between forms |
| B | yes — same | Digest is over signatures from A |
| C | yes — same | Renaming runs on top of byte_key namespaces |
| D | yes — same | Bisim respects builder's identity choices |
| E | yes — same | Iso respects node labels |
| F | yes — same; with caveat | If builder collapses, F works as-is |
| G | yes — same | Same as A |
| H | yes — same | Same as A |

**The 7a fix is universal: it lives in
`_byte_keys_for_resource(resource)` (`ScheduleCapture.py:1018-1052`)
and in the latest-writer resolver's keying strategy. The fix is
independent of which cross-graph approach we adopt for 7b.**

The cleanest 7a fix today (post-c70 `Register` abstraction):
- `Register.from_rocisa(rc)` already produces one `Register` per
  rocisa container with a uniform `(reg_type, name|None, base, count)`
  shape.
- However, numeric and symbolic forms still produce two distinct
  `Register`s: numeric `(reg_type='v', name=None, base=8, count=4)`
  vs symbolic `(reg_type='v', name='ValuA_X0_I0', base=0, count=4)`.
- The 7a fix is to add a **per-graph resolution table** that maps
  symbolic name roots to numeric base indices when both are seen
  within one graph build. This was the c70/d0xd direction. Per the
  bead's reframe, that direction was rejected for the cross-graph
  case but **the intra-graph case still needs a name resolution
  step**, because no topological hashing can synthesize the fact
  that `vgprValuA_X0_I0` and `v8` are the same register if the
  builder doesn't compute that mapping from kernel-writer state.

**Intra-graph fix — OPEN DESIGN QUESTION.** This is now Open Question
#6 (see §6). The kernel-writer state has the mapping (`regName.name →
allocated index`) at the moment the builder forms an edge, but reading
that state at edge-formation time is the c70/d0xd direction the user
just rejected. Several alternatives exist; this memo does not
pre-recommend one. They are enumerated under Open Question #6 below.

---

## 5. Migration story

Today's `compare_graphs`:

```python
ref_keys = reference.edge_keys()         # uses identity (with rendered text)
subj_keys = subject.edge_keys()
missing  = ref_keys - subj_keys
for key in missing:
    diagnose_missing_edge(ref_edges_by_key[key], subject)
```

After the recommended migration (B + F):

```python
if reference.topology_digest() == subject.topology_digest():
    return []                            # FAST PATH
diff_edges = reference.localize_diff(subject)   # F descent
return [f for e in diff_edges
        for f in diagnose_missing_edge(e, subject)]
```

**Code that gets replaced:**
- `DataflowGraph.edge_keys()` (l.471-474) — replaced by
  `topology_digest()` + `localize_diff()` methods on `DataflowGraph`.
- `_identity_for(inst, body_label, category)` (l.772-802) — class_tag
  and loop_index stay; the third element changes from
  `_canonical_render(inst)` to a structural signature
  `(role, position, byte_keys_tuple)`. Probably keep the function
  name; rename the third element semantically.
- The "data-flow node identity sets differ" pre-check inside
  `compare_graphs` (l.2358-2399) becomes "structural node sets
  differ" with the same shape — it remains valuable for fast
  detection of capture-pipeline bugs.
- `compare_graphs` body (l.2401-2414) — rewritten as above.
- `diagnose_missing_edge` — input contract change: takes a
  `(role, position)` lookup of the subject node, not an identity
  lookup. ~10 lines change at the top; the rest of the classifier
  is intact.

**`Failure` class hierarchy:** survives unchanged. All
`FailureNodeLabel`s come from `cms_node_label` which is already
register-name-free.

**Test coverage update estimate.**

- 21 tests in `test_dataflow_graph_comparison.py`. Of these:
  - `test_symbolic_and_numeric_for_same_logical_reg_unchanged`
    (l.499-516) — the 7a limitation pin. Will be **flipped** to
    assert the new behavior (identical structural identity for
    same physical reg in symbolic vs numeric form). 1 test.
  - Render-string identity tests (`test_identity_is_assembly_text`,
    `test_comment_differences_dont_change_identity`,
    `test_whitespace_normalization`,
    `test_mixed_symbolic_numeric_registers_in_same_inst`,
    `test_category_overrides_isinstance_class_tag`) — rewrite to
    pin the new structural identity instead of the rendered text.
    5 tests.
  - Comparison + diagnosis tests (the rest, ~15 tests) — most should
    be unchanged because they exercise observable diagnostic output
    (Failure classes, formatted messages), not the matching surface.
    Estimate ~3-5 of them touch identity tuples directly and need
    structural-identity updates.
- ~35 tests in `test_dataflow_graph_builder.py` — most exercise edge
  formation; structural changes to `_byte_keys_for_resource` (the
  7a fix) may affect ~5-10 tests that pin specific edge sets.
- ~58 tests in `test_ScheduleCapture.py` — `_canonical_render` is
  used here; the function may be retained for diagnostic purposes
  (showing humans the rendered text) even if no longer used for
  matching. Estimate ~3-5 tests need updating.
- Other dataflow-test files (`test_dataflow_graph_scc.py`,
  `_lcc.py`, `_barriers.py`, `_register_gaps.py`,
  `_phantom_edges.py`, total ~115 tests) — almost all exercise
  observable behavior and should pass unchanged. Estimate ~5-10
  need updating for structural-identity changes that propagate
  through edge formation.

**Total updated tests: ~25-40 of 653 unit tests** (rough range
4-6%). All updates are mechanical: replace asserts on rendered
text with asserts on structural signatures, or update fixture
constructions that depend on the old form-distinction behavior.

---

## 6. Recommendation

**Adopt B + F**: cache a top-level structural digest on each
`DataflowGraph` for the O(1) fast-path equality test, and a
per-node Merkle hash table for descend-on-mismatch localization.
F's mismatch output is formatted with role-pair grouping (the
mechanism formerly called Approach H — now subsumed into F's
output layer rather than tracked as a separate approach).

### Why this hybrid

1. **99%-identical fast path is the dominant axis.** B is the only
   approach with a true O(1) on-match return after warm cache; F
   matches it. No other approach has a free short-circuit.

2. **Mismatch localization survives.** F's per-node hash descent
   pinpoints the exact divergent producer/consumer in O(V) without
   re-walking edges; once located, the existing
   `diagnose_missing_edge` classifier runs unchanged.

3. **`Failure` hierarchy survives.** All current diagnostic
   `Failure` classes are register-name-free at their formatting
   surface (they consume `FailureNodeLabel`s from
   `cms_node_label`). The matching surface is the only thing that
   changes.

4. **Intra-graph (7a) is solvable independently.** The 7a fix lives
   in the builder's `_byte_keys_for_resource` and the latest-writer
   resolver's keying strategy — independent of which cross-graph
   approach we adopt. The choice of mechanism for the 7a fix is left
   open (see Open Question #6); the cross-graph 7b recommendation
   (B+F) does not depend on it.

5. **Implementation cost is bounded.** ~250 LoC plus the 7a builder
   fix; touches `compare_graphs`, `_identity_for`,
   `DataflowGraph`, `_byte_keys_for_resource`. No new dependencies
   (BLAKE2b is in `hashlib`).

6. **Determinism is straightforward.** BLAKE2b is byte-stable
   across Python versions and runs; structural signatures are
   built from sortable tuples; no `hash()` builtin.

### Open questions for the user

These need a yes/no before implementation lands:

1. **Should the structural identity tuple include `loop_index`?**
   Today's `_identity_for` does (it's the second element). Topology
   under the new scheme keys on `(role, position)`, and `position`
   already encodes `loop_index` as its first lex component. If
   `loop_index` stays in the identity tuple, cross-body register
   reuse (same role+vmfma+sub but different body_label) stays
   distinct. If removed, two structurally-identical positions in
   ML-1 vs ML collide. **Recommended: keep loop_index.**

2. **Should we keep `_canonical_render` for diagnostics?** The
   rendered string is currently the *only* way to show a human
   "here is the exact instruction that differs." Even with
   structural matching, we still want the rendered text for the
   `Failure` formatter. Keep `_canonical_render` as a display-only
   helper, drop it from the matching path. **Recommended: keep for
   display.**

3. **How to surface "structurally identical but rendered text
   differs"?** Once topology matching is in place, two graphs may
   match topologically but their `_canonical_render` strings differ
   (e.g., genuinely different register allocation between subject
   and reference). Three options:
   - (a) Treat as equal silently. Information is lost; differences
     in register allocation are invisible.
   - (b) Treat as equal with a warning summarizing the rendered-text
     diff at the validator boundary.
   - (c) Treat as equal only when registers match symbolically; flag
     as "topology equal but allocation differs" otherwise.
   **Recommended: (b)** — equal means equal; render diffs become a
   non-blocking diagnostic.

4. **Per-class digest or per-graph digest?** Today the bead's reframe
   says `(src_role, sink_role, edge_kind, byte_offset)` is the edge
   key. Should the digest be over this tuple, or over a richer
   tuple including position? **Recommended: include position** —
   without it, two PackA0 instructions at different vmfma_indices
   would collide in the digest, hiding real reorder bugs.

5. **What should `diagnose_missing_edge` do when an edge is
   "missing" because subject's structural identity for the
   producer is at a different position than reference's?** Today
   identity-mismatch raises `CaptureConsistencyError` early. With
   structural matching, "subject has the producer at vmfma=14 but
   reference has it at vmfma=15" becomes representable. Is that an
   `OrderInvertedFailure` (yes if the consumer relationship
   inverted), a new `PositionShiftedFailure` (no — the bead
   explicitly forbids new diagnostic shapes without need), or
   simply absorbed into `OrderInvertedFailure` semantics?
   **Recommended: absorb into existing classes**; do not invent a
   new Failure type unless a real kernel triggers it.

6. **OPEN — How should the intra-graph 7a builder fix work?** The 7a
   problem (symbolic and numeric references to the same physical
   register live in different byte-key namespaces inside one graph)
   needs a builder-side fix. The recommendation is deliberately
   silent on which mechanism. Candidates:

   - **(a) Builder consumes kernel-writer state at edge-formation
     time.** The builder reads `regName.name → allocated index` from
     the writer state and rewrites symbolic references to numeric in
     the byte-key. **Previously rejected** in the c70/d0xd reframe;
     listed here for completeness.
   - **(b) Local first-write-wins canonicalization within the
     captured body.** No kernel-writer state is consulted. Walk the
     captured body in emission order; the first write that touches a
     given physical-register byte-range assigns a fresh local ID, and
     every subsequent reference to a name or numeric form covering
     that byte-range adopts the same local ID. Self-contained inside
     the capture; no out-of-band state.
   - **(c) Container-instance identity.** Use the `id()` of the
     rocisa container as the namespace key. Already noted in §1
     under Approach F as unavailable in practice — distinct
     constructor calls (`vgpr("ValuA_X0_I0", 4)` vs `vgpr(8, 4)`)
     produce two distinct Python objects, so identity-based keying
     gives two different keys for the same physical register. Listed
     for completeness; does not solve 7a on its own.
   - **(d) Deferred resolution at edge-formation by checking earlier
     byte-range writes in the same body.** Rather than rewrite
     names, the edge builder asks "is there a prior write inside this
     body to any byte-key that overlaps the byte-range I am about to
     read?" If yes, that prior write is the producer regardless of
     whether its container was symbolic or numeric. Producer
     resolution becomes a byte-range search; names are never
     normalized.

   **No tentative recommendation.** The choice is left to the user.

7. **OPEN — When is the cached digest computed?** The B/F
   recommendation depends on a digest cached on `DataflowGraph`. The
   memo doesn't pin down when. Three options:

   - **(a) Eager at build end.** Every graph pays one O(E) digest
     cost at construction whether it is ever compared or not. After
     that, every comparison is O(1).
   - **(b) Lazy on first comparison.** The first comparison still
     pays full O(E) to compute and cache; subsequent comparisons of
     the same graph are O(1). Graphs that are built but never
     compared pay nothing.
   - **(c) Incremental during construction.** Every `add_edge`
     updates a running hash (BLAKE2b supports incremental update);
     the digest is always available. `add_edge` becomes slightly
     more expensive (one hash update per edge), but no separate
     finalization pass is needed.

   **Tentative recommendation: (a) eager at build end**, given the
   bead's 99%-identical assumption — graphs almost always end up in
   a compare call inside the validator pipeline, so paying O(E) once
   at build is virtually free relative to the comparator workload.
   This is tentative pending user decision.

### Estimate

- **Implementation:** ~250 LoC topology-equivalence + ~150 LoC for
  the intra-graph 7a builder fix = ~400 LoC across `CMSValidator.py`
  and `ScheduleCapture.py`. Single bead, single PR.
- **Tests updated:** ~25-40 of 653 unit tests (~4-6%).
- **No new dependencies** (BLAKE2b is in `hashlib`).
- **Failure hierarchy: unchanged.**
- **Backward-compatibility shims: none** (per the bead's no-fallback
  directive).
