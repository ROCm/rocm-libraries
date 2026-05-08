# §7.3 Cross-Subiter ALU-Producer Suppression — Design Discussion

> Bead: `rocm-libraries-uqoz` (this memo). Sibling bead: `rocm-libraries-bwfr`
> (root-cause investigation of the false positive itself). This memo is the
> design-side; bwfr is the causal side. They overlap on "should this exist at
> all" but I am scoped to "given that it exists, is the gate the right
> *shape*."
>
> No code is changed by this memo. All recommendations are for the user to
> decide on.

---

## 1. Summary

The §7.3 carve-out at `CMSValidator.py:2532-2546` is doing real work — it
prevents a known false positive from firing on the TF32 / pack-pipelined
schedules — but it is implemented in the worst possible place: it is a
**silent skip in the classifier**, gated on a category-coupled producer
predicate (`_is_alu_producer`), whose comment claims it "mirrors the
same-subiter gate `_classify_edge_coverage` uses in within-graph mode" — a
claim that is **false in detail**. The within-graph mode does not have a
"same-subiter" gate; it has a *blanket* ALU-producer exemption and an
explicit comment that within-graph order inversion is impossible by
construction. The two gates do not mirror each other; they happen to have
the same effect on this one shape because the within-graph mode never reaches
the predicate.

My two recommendations, in priority order:

1. **Tighten the predicate from "ALU producer" to "ALU producer of the
   specific Pack→MFMA cross-subiter shape we know about"**, and convert the
   silent skip into a positive `# pragma: cms-pipelined` assertion that
   records *why* this edge was suppressed (so a regression in the resolver
   that broadens which edges hit this gate becomes visible). The current
   predicate is wider than the documented justification.
2. **In the longer term, move the suppression out of the classifier
   altogether and into the resolver layer** by making the per-byte
   latest-writer treat symbolic-vgpr cross-subiter Pack writes as
   non-aliasing edges in the first place. The classifier should not be in
   the business of knowing that one specific dataflow shape is
   "pipelining intent." That belongs to whoever forms the false edge
   (bwfr's territory).

Neither recommendation requires immediate action. The question for the user
is which of the two directions to invest in first.

---

## 2. The current shape

### 2.1 The gate

`CMSValidator.py:2521-2553`:

```python
# Phase 1 — gating: order check, default schedule as canonical reference.
ref_p = ref_edge.producer
ref_c = ref_edge.consumer
if p_node.body_label == c_node.body_label:
    default_p_before_c = ref_p.position < ref_c.position
    subj_p_before_c = p_node.position < c_node.position
    if default_p_before_c and not subj_p_before_c:
        # Cross-subiter ALU-producer edges are a known false-positive
        # source: a PackA3 (subiter 3) writes a symbolic vgpr that an
        # earlier-subiter MFMA reads under the same symbolic name. The
        # default schedule emits all Packs before all MFMAs (linear
        # within-body); CMS pipelines so subiter-N+1's Pack issues after
        # subiter-N's MFMA — the order inversion across subiters is
        # legitimate pipelining, not a real reorder of a same-subiter
        # dependency. Mirrors the same-subiter gate
        # _classify_edge_coverage uses in within-graph mode.
        nmps = subj_graph.num_mfma_per_subiter
        if (_is_alu_producer(p_node)
                and _node_subiter(p_node, nmps)
                    != _node_subiter(c_node, nmps)):
            return []  # cross-subiter pipelined dependency — legitimate
        return [OrderInvertedFailure(...)]
```

The exact predicate that triggers the silent skip is the conjunction:

  * **Same body** — `p_node.body_label == c_node.body_label`
  * **Inversion** — `ref` orders producer-before-consumer; `subj` orders
    consumer-before-producer.
  * **`_is_alu_producer(p_node)`** — see `CMSValidator.py:1842-1894`.
    Includes all `Pack*` (except 4x4 PackMFMAs which `_is_mfma_pack_producer`
    diverts to the MFMA branch), plus any non-LR/LW/GR/MFMA rocisa instance.
  * **Producer subiter ≠ consumer subiter** — per
    `_node_subiter` (`CMSValidator.py:737-753`). Subiter is parsed from
    category trailing digits for non-MFMA, or `vmfma_index //
    num_mfma_per_subiter` for MFMA.

When all four hold: `return []`. No failure, no warning, no log line.

### 2.2 Source-comment rationale (recap)

> *"A PackA3 (subiter 3) writes a symbolic vgpr that an earlier-subiter MFMA
> reads under the same symbolic name. The default schedule emits all Packs
> before all MFMAs (linear within-body); CMS pipelines so subiter-N+1's Pack
> issues after subiter-N's MFMA — the order inversion across subiters is
> legitimate pipelining, not a real reorder of a same-subiter dependency.
> Mirrors the same-subiter gate `_classify_edge_coverage` uses in
> within-graph mode."*

That is two factual claims: (a) the inversion is benign because pipelining,
(b) within-graph mode mirrors this. Claim (a) is the substance of bwfr's
investigation. Claim (b) I can falsify directly — see Topic 2 below.

---

## 3. Topic 1 — Scoping correctness

### 3.1 What the predicate covers vs. what the rationale justifies

The rationale specifically describes one shape: **a `Pack*` producer in
subiter N+1 that is read by an MFMA in an earlier subiter**. The predicate
is wider than that justification in two ways:

* **Producer side: "ALU producer" includes more than `Pack*`.**
  `_is_alu_producer` returns True for any non-MFMA, non-LR, non-LW, non-GR
  rocisa instruction. So a vector ALU op (e.g. a `VAddU32` writing a
  symbolic vgpr that gets read by an earlier-subiter MFMA), an m0 setter,
  a GRInc, a CVTPack feeding a non-MFMA consumer — all of these will hit
  the silent-skip path when the order is inverted across subiters. The
  rationale does not justify any of these.
* **Consumer side: no constraint at all.** The predicate doesn't check
  what the consumer is. The comment talks about MFMA consumers
  specifically. A Pack→Pack cross-subiter inversion would also hit the
  silent skip.

I do not have evidence that any of these *additional* shapes occur in real
captures today. But the predicate has no reason not to fire on them; the
gate would happily mask a non-pipelining inversion if one ever appeared.

### 3.2 Could this mask a real bug? Concrete scenarios

**Scenario A — non-Pack ALU writer/MFMA reader cross-subiter inversion.**
Suppose CMS reorders a `v_xor_b32 v200, vS, vT` (ALU, no category=Pack)
that writes v200, past an MFMA in an earlier subiter that reads v200. The
default puts the XOR before the MFMA; CMS pipelines and puts the MFMA
first. This is structurally the same shape as the documented case but
with a non-Pack ALU producer. The gate fires. Whether the inversion is
benign is unknown — there is no `Pack*` semantics to lean on; this is
just "an ALU writer that happened to land on a vgpr the MFMA reads
later." If the reorder is wrong, the validator now hides it.

**Scenario B — LR producer feeding an earlier MFMA across subiters.**
LR is *not* an ALU producer in `_is_alu_producer` (the function explicitly
excludes `_is_lr(inst)` at `:1888`). So an LR cross-subiter inversion
would still fire `OrderInvertedFailure`. This is the right behavior;
flagging it because the user asked the question. The current scoping does
*not* mask LR producers.

**Scenario C — GR producer.** Same as LR: GR is excluded from
`_is_alu_producer`. Real GR cross-subiter inversion would still fire. Good.

So the only meaningful exposure today is **non-Pack ALU producers**. That
is a small risk surface but not zero.

### 3.3 The "same symbolic name" caveat

The rationale's load-bearing word is **symbolic**: "writes a symbolic vgpr
that an earlier-subiter MFMA reads under the same symbolic name." The
implication is that the false edge arises from *naming*, not from a real
dataflow reuse. The per-byte latest-writer resolver
(`CMSValidator.py:3098-3153`, see VALIDATOR_DESIGN.md §7.6) doesn't
actually look at symbolic names — it operates on rendered byte addresses.
So either (a) the symbolic names resolve to the *same physical* vgpr (in
which case the dependency is real and the suppression is wrong), or
(b) the symbolic names resolve to *different physical* vgprs (in which
case the resolver should have produced no edge in the first place — the
edge is an artifact and the suppression is treating a symptom). The
predicate doesn't distinguish these cases.

bwfr's investigation will tell us which it is. Until then, the
recommendation is: **even if (a)** turns out to be the answer, the gate
should fire on Pack→MFMA cross-subiter specifically, not on the wider
ALU-producer set.

### 3.4 Recommendation for Topic 1

Tighten the predicate to:

```python
if (_is_pack_producer(p_node)         # not _is_alu_producer
        and _is_mfma_producer(c_node)  # add consumer-side gate
        and _node_subiter(p_node, nmps) != _node_subiter(c_node, nmps)):
    return []
```

This gives up nothing the rationale claims to need, and removes the
hypothetical risk of masking non-Pack ALU inversions. The user might want
to keep the gate as-is if the broader shape is in fact occurring in
captures and is also believed benign — but that should be an explicit
decision with evidence, not a side-effect of `_is_alu_producer`'s breadth.

---

## 4. Topic 2 — Composition with within-graph mode

The source comment claims this gate "Mirrors the same-subiter gate
`_classify_edge_coverage` uses in within-graph mode." Let me verify
against `CMSValidator.py:2864-2960`.

### 4.1 What `_classify_edge_coverage` actually does

At `CMSValidator.py:2880-2885`:

```python
# Phase 1 — same-body order check is no longer needed here: Sub-B's
# per-byte latest-writer resolver only emits edges where producer is
# before consumer in stream order. Within-graph OrderInverted detection
# is therefore impossible by construction; the cross-graph classifier
# in diagnose_missing_edge owns OrderInverted detection (with default
# positions for diagnostics).
```

And at `CMSValidator.py:2957-2960`:

```python
# ALU-as-producer: results are immediately visible; no wait counter
# applies. Within-graph order inversions were already handled above.
if _is_alu_producer(p_node):
    return []
```

That is **not** a same-subiter gate. It is a **blanket ALU-producer early
return**, *unconditional* on subiter, scoped by the higher-level claim
that order inversion can't appear here in the first place.

### 4.2 So is the comment wrong?

Yes, in the way that matters. The two gates have the same predicate name
in their early-return arm (`_is_alu_producer`), but:

* Cross-graph (`:2543`): `_is_alu_producer` AND **subiter mismatch** →
  silent skip. With subiter match → fire `OrderInvertedFailure`.
* Within-graph (`:2959`): `_is_alu_producer` → silent skip. No subiter
  check. Comment justifies the lack of a check by pointing upstream
  ("within-graph order inversion is impossible by construction").

Stated directly: **the within-graph mode does not need a subiter gate
because, in that mode, the resolver guarantees no inversion exists to
gate.** The cross-graph mode does not have that guarantee (different
graphs, different stream orderings), so it added a gate. The two are
different mechanisms for different problems; calling it a "mirror" obscures
that.

### 4.3 What this composition implies

There are two follow-on observations:

1. **The comment in `:2540-2541` is misleading and should be rewritten**
   regardless of any other change. The user should know the within-graph
   mode is not validating "this is benign cross-subiter pipelining" — it is
   skipping ALU producers wholesale. There is no within-graph mirror.
2. **If the false-positive root cause (bwfr) turns out to be in the
   resolver**, fixing it there would automatically fix both sites,
   because both rely on the same upstream invariant (no spurious
   cross-subiter Pack edge in the first place). Conversely: if the
   resolver is the right fix, the cross-graph gate becomes dead code, not
   a "mirror" of anything.

### 4.4 Cross-graph also has a duplicate ALU-bail-out

For completeness: cross-graph mode *also* has the unconditional ALU-producer
early return, separately, at `CMSValidator.py:2672-2677`:

```python
# ALU-as-producer (scalar/vector ALU, GRInc, m0 setters): result is
# immediately visible to the next issued instruction; no SWaitCnt drain
# applies. Phase 1 already classified any order inversion; nothing else
# to verify.
if _is_alu_producer(p_node):
    return []
```

So in cross-graph mode, an ALU producer with a *same-subiter* inversion
(`default_p_before_c and not subj_p_before_c`, same subiter) → `OrderInverted`
is fired by the Phase-1 branch. An ALU producer with a *cross-subiter*
inversion → silent skip. An ALU producer with **no** inversion (or order
preserved) → falls through to `:2676` and silently returns []. The
classifier therefore has three exits for ALU producers depending on order
relationship; only one of those exits emits a failure. Worth being aware of
when reasoning about coverage.

### 4.5 Recommendation for Topic 2

Independent of the wider redesign, **rewrite the comment** at
`CMSValidator.py:2540-2541` to say:

> "In within-graph mode, `_classify_edge_coverage` cannot see this case
> because the resolver guarantees no inversion within a single graph
> (`CMSValidator.py:2880-2885`). This cross-graph gate is the only place
> the suppression is enforced."

The "mirrors" wording invites a future maintainer to assume the two checks
are coupled and break one when changing the other.

---

## 5. Topic 3 — Positive assertion vs silent skip

### 5.1 Why "silent return []" is a code smell here

Today the classifier hands every other end-state a typed `Failure` or a
positive cover-by-construction reason (Phase 2 wait-coverage classifies
positively). The §7.3 carve-out is the only place the classifier returns
`[]` for a *real* missing edge with no record. From a debuggability
standpoint, this is the worst kind of suppression — if a bug ever causes
this gate to mask something it shouldn't, there is nothing to grep for in a
failing-build artifact.

### 5.2 What a positive-assertion form would look like

A positive assertion needs to express the precondition that makes the
suppression sound. Concretely, the pipelining narrative says: *"CMS chose
to issue subiter-N+1's Pack after subiter-N's MFMA; the MFMA's read of
the symbolic vgpr is satisfied by the corresponding write from a prior
loop iteration."*

To turn that into a check, the classifier would need to know:

* The MFMA reads vgpr v_X.
* In the prior iteration (or earlier in this iteration), there exists a
  write to v_X that the resolver pairs with this MFMA's read.
* The Pack we are now silently skipping writes a *future* value of v_X
  that the *next* iteration's MFMA will read.

In dataflow terms: assert that the MFMA's read has at least one *other*
producer edge in `subj_graph` (the prior-iteration write), and that the
Pack producer in question has at least one *other* consumer edge in
`subj_graph` (the next-iteration MFMA read). If both hold, the suppression
is justified by positive evidence; if not, the gate has fired without
the structural shape it claims to know about.

In code (sketch — not for inclusion):

```python
prior_writers = [e for e in subj_graph.edges
                 if e.consumer.identity == c_id
                 and e.producer.identity != p_id
                 and _is_alu_producer(e.producer)]
later_readers = [e for e in subj_graph.edges
                 if e.producer.identity == p_id
                 and e.consumer.identity != c_id
                 and _is_mfma_producer(e.consumer)]
if prior_writers and later_readers:
    return []  # positively classified pipelined dependency
# Otherwise the silent-skip premise is unmet — fire the failure.
return [OrderInvertedFailure(...)]
```

### 5.3 Pros / cons

**Pros:**
* Defeats the only gap in the "no silent ignores" contract VALIDATOR_DESIGN.md
  §1 establishes (every Failure is typed; every fall-through raises). The
  validator currently makes this one exception.
* Catches regressions where the resolver starts producing the same gate
  shape for non-pipelining reasons.
* Documents the suppression in code, executable form (the predicate IS
  the documentation).

**Cons:**
* More expensive — two graph-edge scans per gate hit. Probably
  immaterial; this gate fires O(packs × subiters) times per kernel.
* Adds a tighter coupling between the classifier and the dataflow graph
  (already coupled, but more so).
* If the positive-evidence check is wrong, the gate may reject benign
  cases the current gate accepts. Migration risk.

### 5.4 Recommendation for Topic 3

I would do this. The "no silent ignores" contract is one of the validator's
strongest design properties; this gate is the lone exception. A positive
assertion brings it back into the fold. The implementation is small, and
even if you eventually move the suppression to the resolver (Topic 4) the
positive form is the right intermediate shape — it makes the gate's
preconditions auditable while the resolver work is in flight.

The user's "should it be an assertion" framing is correct.

---

## 6. Topic 4 — Layer placement

### 6.1 The current placement and why it is wrong-shaped

The classifier's job is to **explain** missing edges, not to **decide**
which dataflow edges should exist. Today's gate is doing both — it sees
that an edge is missing and decides "no, that edge shouldn't have been
there in the first place." That decision belongs upstream.

### 6.2 The resolver-layer alternative

`build_dataflow_graph`'s phase 2 (`CMSValidator.py:3098-3153`) walks data-
flow nodes in stream order and maintains a `latest_writer` map keyed by
byte. Reads emit one edge per distinct prior writer. If the symbolic-vgpr
question turns out to be "different physical registers, but the renderer
makes them look the same to the per-byte resolver" (bwfr's case (b)), the
fix is to make the resolver disambiguate the cases such that no false edge
is formed in the default-side graph. Then `compare_graphs` would never see a
"missing" edge to suppress; the classifier wouldn't need a gate; the
within-graph mode wouldn't need its `Within-graph order inversion is
therefore impossible by construction` reassurance to extend to a special
case.

### 6.3 What invariant would the resolver need to maintain?

> **"Every edge `latest_writer` emits names a real physical-register
> producer→consumer dependency, with no aliasing across symbolic-name
> reuse cycles."**

In practice, that means: when the renderer emits two `Pack` writes whose
target vgprs spell the same symbolic name across subiters, the resolver
must distinguish them by some other key (the SchedulePosition's loop_index
+ subiter, or an explicit pipeline-cycle counter, or the rendered numeric
vgpr after symbolic resolution). Today the resolver works on rendered byte
addresses, so this is "make the rendered byte address actually
disambiguate the two writes." The byte-key is correct in principle; the
question is whether the rendering step that feeds it is.

### 6.4 Cost

Three concerns:

* **Scope.** This is a resolver change, not a classifier change. The
  resolver is one of the most heavily-tested components. Touching it
  to fix a classifier-level symptom is high-blast-radius.
* **Cross-cutting impact.** Many tests synthesize fixtures with
  `vgpr(N)` literals and rely on byte-key aliasing as a cheap way to
  express "same register". Changing the disambiguation key would force
  those fixtures to opt in to the new shape (or break).
* **Over-fit.** If the false positive turns out to be just one specific
  shape (e.g. only the symbolic `valuA_T0_I0` family), a resolver-layer
  fix may be over-investing. A scoped fix in the renderer that emits the
  Pack-writes' vgpr indices explicitly may be cheaper.

### 6.5 Recommendation for Topic 4

Conditional on bwfr. If bwfr concludes the false edge is a
*resolver/renderer artifact*, move the suppression to the
resolver/renderer layer and delete this classifier gate entirely (no
positive assertion needed because there is nothing to assert about). If
bwfr concludes the false edge is a *real dataflow shape that we
intentionally pipeline*, keep the suppression in the classifier but
upgrade it to the positive-assertion form (Topic 3) and tighten the
predicate (Topic 1).

The two recommendations are complementary, not exclusive. Topics 1+3 are
"do this now"; Topic 4 is "do this once bwfr lands."

---

## 7. Other observations

### 7.1 The `nmps == 0` degeneration

`_node_subiter` (`CMSValidator.py:737-753`) collapses MFMA subiters to 0
when `num_mfma_per_subiter` is 0 (test fixtures that don't set it). The
docstring acknowledges:

> *"the OrderInverted gate then degenerates to 'fire on any same-body
> stream-position inversion'."*

This means the gate's *suppression* turns off in test fixtures, so test
fixtures will see `OrderInvertedFailure` for cross-subiter Pack inversions
that production code would silently skip. That's actually fine for unit
tests — it makes the failure mode visible — but it means the gate's
protective behavior is invisible to the test suite. **A test that
specifically exercises the suppression path needs `num_mfma_per_subiter ≥
1` set on the FourPartCapture.** I did not find such a test in
`test_dataflow_graph_comparison.py` — the `TestGenuineOrderInversionStillDetected`
test (`:1037`) uses `LRA0` (subiter 0 for both) and a pure ALU pair, and
it tests the *non-suppressed* inverted case. There is no positive test
asserting the suppression itself fires correctly. **This is a test-coverage
gap** worth filing as a follow-up.

### 7.2 The `_is_alu_producer` Pack carve-out is itself complex

`_is_alu_producer` (`CMSValidator.py:1842-1894`) special-cases TF32 4x4
PackMFMAs out of the ALU path and into the MFMA path via
`_is_mfma_pack_producer`. The §7.3 gate inherits this carve-out, which
means a 4x4 PackMFMA cross-subiter inversion would NOT be suppressed; it
would fall through to `OrderInvertedFailure`. This is presumably intended
(PackMFMAs have real finish-cycle constraints that the pipelining
narrative doesn't apply to). Worth documenting: the gate is not "all
Pack* producers", it is "Pack* producers minus PackMFMA-via-`_is_mfma_pack_producer`".

### 7.3 `compare_graphs` does not check the reverse direction

VALIDATOR_DESIGN.md feedback line 14 already notes that `compare_graphs`
computes `missing_keys = ref_keys - subj_keys` only; the reverse is not
checked. This means a CMS-only edge (one that exists in subj but not in
ref) is silently accepted. If the symbolic-vgpr aliasing produces an
*extra* edge in subj rather than (or in addition to) a missing edge in
ref, the §7.3 gate isn't even the right place to look — `fo40` (the bead
about the symmetric check) is. Flagging this overlap; bwfr may benefit
from the symmetric direction too.

### 7.4 The "Legitimate-CMS-reorder" branch above the gate

`CMSValidator.py:2494-2519` — the wx9.3 §6.1 branch — already filters out
missing-key cases that turn out to be present in subj under the same
producer/consumer/edge-kind/byte-offset keying but a different
SchedulePosition. **By the time the §7.3 gate runs, we have already
established that no edge between this producer.identity and
consumer.identity exists in subj at all.** That's worth stating explicitly
in the gate's comment — the §7.3 case is genuinely a missing edge, not a
relocated one. The comment as written might lead a reader to think
"missing" means "moved", which the legitimate-reorder branch handles
separately.

### 7.5 Telemetry would be useful

If the gate stays in its current shape (silent return []), at minimum a
debug-level log line counting how often it fires would let us confirm in
real captures whether the shape we're justifying is the *only* shape that
hits this gate. Today there is no way to know from outside whether the
gate is firing 0 times, 5 times, or 500 times per kernel. A counter on
`DataflowGraph` (or a `failures` sibling field listing "suppressed
classifications") would close this gap cheaply.

---

## 8. Open questions for the user

These are the concrete decisions the discussion needs from you:

1. **Is the breadth of `_is_alu_producer` in the gate predicate
   intentional, or is the gate intended to apply only to the documented
   `Pack*`-cross-subiter shape?** (Topic 1.) If intentional, what other
   shapes does it cover that you've seen in real captures?
2. **Should §7.3 be the only "no silent ignore" exception in the
   validator, or should the gate be converted to a positive-assertion
   form so the validator's "either knows or admits it doesn't know"
   contract holds uniformly?** (Topic 3.) The trade-off is one extra
   graph scan per gate hit vs. continuing to allow one quiet exception.
3. **If bwfr concludes the root cause is in the resolver/renderer, do
   you want to move the suppression upstream and delete this gate
   entirely?** (Topic 4.) Or are you comfortable carrying the gate
   indefinitely as a known carve-out?
4. **Is there appetite for a positive test of the suppression path
   itself** (a test that builds a fixture with `num_mfma_per_subiter ≥ 1`
   and a cross-subiter Pack→MFMA inversion, asserting the gate fires
   and `compare_graphs` returns `[]`)? (Observation 7.1.) Today the
   suppression has no direct test coverage that I can find.
5. **Should the comment at `CMSValidator.py:2540-2541` claiming the gate
   "mirrors the same-subiter gate `_classify_edge_coverage` uses in
   within-graph mode" be rewritten now**, independent of any structural
   change? (Topic 2 / 4.5.) The claim is misleading; the within-graph
   "gate" is actually a blanket ALU-producer skip plus an upstream
   no-inversion guarantee.

---

## 9. Cross-references

* Sibling bead `rocm-libraries-bwfr` — root-cause investigation of the
  false-positive itself. This memo is design-side; bwfr is causal-side.
  If bwfr concludes the resolver/renderer is at fault, Topic 4 becomes
  the dominant recommendation. If bwfr concludes the false positive is
  intrinsic to how CMS pipelines, Topics 1+3 dominate.
* Bead `rocm-libraries-fo40` — the symmetric-direction `compare_graphs`
  check. Possibly relevant if the false edge appears in subj rather than
  in ref. Flagged in §7.3 of this memo.
* Bead `rocm-libraries-o0ei` — pair-specific quad-cycle helper dispatch
  order. Tangentially related: the `_is_alu_producer` exemption breadth
  shows up there too (PackMFMAs have to run before the ALU early-return
  claims them). Same root cause — `_is_alu_producer` is doing more
  routing work than its name suggests.
* `VALIDATOR_DESIGN.md` §7.3 (lines 373-388) — the section this memo
  expands on.
* `VALIDATOR_DESIGN.md` §2 goal 1 (lines 60-66) — where the suppression
  is enumerated as the validator's only documented carve-out.

---

*End of memo. No code changes accompany this document.*
