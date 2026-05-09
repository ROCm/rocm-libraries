# compare_graphs Symmetry Investigation (rocm-libraries-fo40)

> Bead: `rocm-libraries-fo40`. Investigator: fo40-investigator (Claude
> Opus 4.7, May 2026). Working tree:
> `/home/alvasile/rocm-libraries/.worktrees/fo40-investigation/`.
>
> Sibling beads: `rocm-libraries-bwfr` (root cause of the cross-subiter
> ALU false positive), `rocm-libraries-uqoz` (design discussion of the §7.3
> carve-out's shape and placement).
>
> No code changes accompany this memo.

---

## 1. Summary

`compare_graphs` (`CMSValidator.py:2422-2505`) computes
`missing_keys = ref_keys - subj_keys` and routes every missing key
through `diagnose_missing_edge`. The reverse direction
(`subj_keys - ref_keys`) is **not** computed. The asymmetry IS
load-bearing: `Tensile/Components/VALIDATOR_DESIGN.md` §4.1 explicitly
declares the default schedule canonical, and the missing-edge classifier
is built around that asymmetry — every Failure subclass except
`OrderInvertedFailure` already presupposes "subj is the candidate, ref
is correct." The probe in `Tests/scratch/probe_fo40_extra_edges.py`
shows that **most extra edges are paired with missing edges by
construction** (the per-byte resolver's mechanics force this), and
those pairs are already absorbed by the wx9.3 §6.1 legitimate-CMS-
reorder branch (`CMSValidator.py:2546-2571`). I found **one** paired
case where the extra carries information the missing side did not
(scenario 1 below: a CMS clobber-swap where the extra contains the
CMS-correct edge while the missing fires `OrderInvertedFailure`), but
the structural fact "subj contains an edge ref does not" is already
implied by `OrderInvertedFailure`'s firing — the symmetric check would
only restate it.

**Recommendation: option 1 (keep the asymmetry, document loudly in §4.1
of `VALIDATOR_DESIGN.md`).** The probe data does not motivate option 2
or 3; the symmetric check would either restate information already
captured by `OrderInvertedFailure` or fire on legitimate CMS-side
reorders that the wx9.3 §6.1 branch was built to absorb (creating a
fresh false-positive surface that the carve-out would have to grow to
suppress). I recommend a documentation patch + a single explicit unit
test that pins the asymmetry as intentional. Effort: ~1 hour. Risk:
zero.

---

## 2. Current behavior

### 2.1 The asymmetric site

`CMSValidator.py:2485-2505`:

```python
ref_keys = reference.edge_keys()
subj_keys = subject.edge_keys()
missing_keys = ref_keys - subj_keys

# Map missing keys back to reference edge objects for diagnosis.
failures = []
ref_edges_by_key = {
    (e.producer.identity[0], e.producer.position, e.src_operand_slot,
     e.consumer.identity[0], e.consumer.position, e.sink_operand_slot,
     e.edge_kind, e.intra_operand_byte_offset): e
    for e in reference.edges
}
for key in missing_keys:
    ref_edge = ref_edges_by_key[key]
    failures.extend(diagnose_missing_edge(ref_edge, subject))
return failures
```

`subj_keys - ref_keys` is never computed; `subj_keys` is read into
local scope only to compute `missing_keys`.

### 2.2 What IS symmetric

The identity-coverage check at `CMSValidator.py:2442-2483` IS
symmetric: it compares the two graphs' DATA-FLOW node identity sets
(LR/LW/GR/MFMA) and raises `CaptureConsistencyError` if either side
has identities the other lacks. Crucially, this is a node-set check,
not an edge-set check, and it's about catching capture-pipeline bugs
(an instruction missing from one capture), not about catching CMS
mis-scheduling.

So the validator's symmetry contract is:

| Layer                    | Direction | Reaction                               |
|--------------------------|-----------|----------------------------------------|
| Data-flow node identity  | both      | `CaptureConsistencyError` (raises)     |
| Edge set: ref - subj     | one-way   | per-edge classify via diagnose_missing_edge |
| Edge set: subj - ref     | unchecked | silent pass                            |

### 2.3 What `validate_edge_wait_coverage` provides indirectly

`validate_edge_wait_coverage` (`CMSValidator.py:2869-2913`) walks every
edge in the CMS graph and demands a covering wait/barrier in the
captured stream. So an extra edge in CMS that lacks a covering wait
WOULD surface — as a `MissingWaitFailure`, `WaitInsufficientFailure`,
or `MissingBarrierFailure`. Two consequences:

1. The structural fact "this edge does not exist in the canonical
   default" never surfaces, but the user still gets a failure if the
   extra edge is unsafe.
2. The structural fact "this edge does not exist in the canonical
   default but IS properly synchronized" is fully silent. If the user
   wants to audit the diff for any reason — e.g. to confirm CMS only
   ever specializes the schedule's timing without changing its
   dataflow shape — they cannot do so today.

### 2.4 Why the missing-side classifier presupposes asymmetry

Reading `diagnose_missing_edge` (`CMSValidator.py:2508-2867`) end to
end, every classifier branch is built around "ref says X must hold;
check if subj satisfies X." Concretely:

- Phase 0 (line 2538): missing producer/consumer in subj → raise.
- Phase 1 (line 2581): same-body order check — uses `ref_p` /
  `ref_c` (default's positions) as the "ground truth" order.
- SCC clobber path (line 2629): looks for an intervening SCC writer in
  subj that explains why the consumer lost its ref-side producer.
- MFMA / CVT / ALU producer branches: structural exemptions, all
  driven by the ref edge's producer category.
- Phase 2 wait-coverage check (line 2737): does subj have a wait that
  drains ref's producer?

There is no "for an edge in subj that ref doesn't have, what should we
say?" branch anywhere. All `Failure` subclasses except OrderInverted
take the producer/consumer from a *known-correct* reference; an extra
edge has no such reference.

---

## 3. Probe — does the asymmetry hide real bugs?

### 3.1 Setup

The probe lives at
`Tests/scratch/probe_fo40_extra_edges.py` (uncommitted; same scratch
convention as bwfr). Four scenarios. All four use the standard
`dataflow_fixtures` builders + a private `_wrap` to fill body parts.
Run with:

```bash
PYTHONPATH=projects/hipblaslt/tensilelite/Tensile/Tests/unit:projects/hipblaslt/tensilelite \
    python projects/hipblaslt/tensilelite/Tensile/Tests/scratch/probe_fo40_extra_edges.py
```

### 3.2 Scenario 1 — clobber-swap (LRA0/LRA1 to same vgpr)

Two LRs both writing v8..v11 (different LDS offsets), one MFMA
reading v8..v11. Default emits LR_A0, LR_A1, SWait, MFMA;
resolver records `latest_writer[v8]=LR_A1`, edge LR_A1→MFMA. CMS
pipelines: LR_A0, SWait, MFMA, LR_A1; resolver at MFMA's read
records `latest_writer[v8]=LR_A0`, edge LR_A0→MFMA.

Probe output:

```
ref edges: 1, subj edges: 1
missing: 1   ('LR', pos(li=1,si=1), 0, 'MFMA', pos(li=1,si=3), 0, ...)
extra  : 1   ('LR', pos(li=1,si=0), 0, 'MFMA', pos(li=1,si=2), 0, ...)
compare_graphs failures: 1   OrderInvertedFailure(LRA1[0]@idx=3 -> MFMA@idx=2)
```

**Interpretation.** The missing-side check correctly identifies that
LR_A1 was reordered past MFMA in subj — `OrderInvertedFailure` fires.
The extra-side edge (LR_A0→MFMA) is the CMS-CORRECT dataflow edge.
The structural fact "CMS introduced LR_A0→MFMA which wasn't in
default" is silently dropped. *But:* the user already knows from the
OrderInverted that LR_A1 was reordered, and the carve-out logic does
the right thing (or, in this exact case, doesn't fire because LRs are
not ALU producers — see `_is_alu_producer` at `CMSValidator.py:1842`).
A symmetric check would emit something like `UnexpectedExtraEdge(LR_A0
@idx=0 → MFMA@idx=2)` which restates "the producer side of the
inversion has shifted." Useful in a debugging report? Possibly. Useful
as a Failure that needs distinct handling? No.

### 3.3 Scenario 2 — same dataflow, swapped LR stream order

Two LRs writing v8..v11 (operand A) and v12..v15 (operand B), one
MFMA reading both. Default and CMS differ only in which LR is at
stream 0 vs stream 1. The resolver pairs each LR with its respective
MFMA-operand by `byte_key`, so both graphs form the *same two
semantic edges* (LR_v8→MFMA[A], LR_v12→MFMA[B]) but with different
`(src_position, sink_operand_slot)` keys.

Probe output:

```
ref edges: 2, subj edges: 2
missing: 2   (paired with extras — same identities, swapped operand slots)
extra  : 2
compare_graphs failures: 0   <-- legitimate-CMS-reorder branch absorbs all
```

**Interpretation.** This is the canonical "CMS just rearranged
unrelated reads" case. The wx9.3 §6.1 legitimate-CMS-reorder branch
(`CMSValidator.py:2546-2571`) recognizes that for each missing key,
the same logical (`producer.identity`, `consumer.identity`,
`edge_kind`, `byte_offset`, `src_operand_slot`, `sink_operand_slot`)
edge already exists in subj, and silently drops the diagnosis with
`return []`. Net: 2 missing keys, 2 extra keys, **0 reported
failures**. The validator correctly says "no problem here."

But notice: if a symmetric check were added, **all 2 extra keys would
fire** — because the legitimate-CMS-reorder branch only runs inside
`diagnose_missing_edge` (driven by missing keys). For each extra key
to be similarly absorbed, an analogous reverse-direction
short-circuit would have to be built. That is one of three cost
items option 2 / 3 would carry.

### 3.4 Scenario 3 — bwfr-mirrored (CMS clobbers same-vgpr)

The probe constructs three Pack writers all targeting v8 plus two
MFMA readers. **Caveat:** in this synthetic setup the three Packs
share an identity tuple (same rendered string `v_mov_b32 v8, v108`)
because the test fixture deduplicates by instruction signature. The
graph collapses three Pack nodes into one. So the scenario as
written exercises a degenerate case (1 Pack, 1 MFMA after
deduplication) and produces 1 missing + 0 extra. Not informative for
the symmetric-edge question.

A faithful version would require distinct rendered strings per Pack,
e.g. `v_mov_b32 v8, v108` / `v_mov_b32 v8, v109` / `v_mov_b32 v8,
v110`. I did not construct that variant because (a) the
identity-collapse behavior is itself worth flagging — see §3.6 — and
(b) scenarios 1 and 2 already cover the structurally interesting
shapes for this investigation.

### 3.5 Scenario 4 — extra-only candidate

Same construction as scenario 3 (three Packs, one MFMA) with the same
identity-collapse caveat. Probe shows 1 missing, 0 extra, 0 failures
— the carve-out at `CMSValidator.py:2585-2598` swallows the missing
edge. Not an extra-only finding; the synthetic fixture cannot
manufacture one without distinct producer identities.

### 3.6 What I could not contrive

I attempted but failed to construct a realistic fixture where:

- subj has a true subj-only edge (i.e. an edge whose
  (`src_role`, `src_position`, `src_operand_slot`, `sink_role`,
  `sink_position`, `sink_operand_slot`, `edge_kind`,
  `intra_operand_byte_offset`) tuple appears in `subj_keys` but not
  in `ref_keys`),
- where the producer node IS in both graphs (identity-coverage
  enforced),
- and where there is NO paired missing edge to the same consumer.

The reason: the per-byte latest-writer resolver assigns each consumer
read **exactly one producer per (read_resource, byte_key)** at any
given stream position. So if subj has an edge X→C for read R, ref
also has *some* edge to C for read R (possibly Y→C, where Y ≠ X).
The two graphs' edges into C are paired by R. The "extra" subj
edge always has a corresponding "missing" ref edge into the same
consumer.

**So the *only* mechanism by which `subj_keys - ref_keys` is
non-empty is via this pairing.** Every extra edge has a missing
sibling. The missing sibling already triggers `diagnose_missing_edge`.
Any classification that the symmetric check could perform on the
extra has, by construction, already been performed (or absorbed) on
the missing side.

(Edge case I did not exhaust: cross-iteration / cross-body edges
where ref's resolver fails to form an edge for one body but subj's
forms one. This would require a body shape difference, which the
identity-coverage check or the `Tests/unit/test_dataflow_graph_comparison.py::TestPerFailureDiagnosis`
existing fixtures already exercise indirectly.)

### 3.7 Hidden-bug surface estimate

Given §3.6, the only hidden-bug surface today is:

- A subj-extra edge whose consumer also has a *correctly-classified*
  producer in subj (so its missing-sibling diagnosis returned `[]`
  via the legitimate-CMS-reorder branch or one of the structural
  exemptions).
- That edge is also covered by a wait/barrier in the captured stream
  (so `validate_edge_wait_coverage` doesn't flag it).
- But the edge represents a real unintended dataflow change — e.g.
  CMS introduced a new producer→consumer relationship the kernel
  writer never authored.

This is a small surface. I cannot construct it synthetically without
a paired missing edge that would already fire something. I cannot
rule out that it occurs in real captures, but bwfr's investigation
of the production failures (10 production-kernel tests, 768 edge
differences total — `CROSS_SUBITER_ALU_FP_INVESTIGATION.md` §3.2)
showed those failures all live on the *default* side, not the CMS
side. That is one data point that the asymmetry is in the right
direction: the resolver clobbers happen on the side that linearizes,
which is the default. CMS's pipelining tends to keep producers and
consumers tighter, with less clobber risk.

---

## 4. Connections to bwfr / uqoz / §7.3

### 4.1 bwfr's resolver-artifact direction

bwfr (`CROSS_SUBITER_ALU_FP_INVESTIGATION.md` §4.2) showed that the
artifact edges in the production scratch-vgpr-reuse case live on the
**default** side: the default schedule emits all Packs before all
MFMAs, the resolver overwrites `latest_writer[v133]` with each new
Pack, and MFMA0's read resolves to PackA1 instead of PackA0 in the
default graph. CMS, by pipelining, keeps the right Pack adjacent to
the right MFMA, so the resolver assigns producers correctly.

The *artifact* lives on the ref side. The *correct* edges live on
the subj side. `compare_graphs`'s `ref - subj` direction surfaces the
artifact (correctly identified by Phase-1 + the carve-out as
"ignore"); the `subj - ref` direction would surface the correct
edges (which would fire as "unexpected extras" if any naive
`UnexpectedExtraEdge` Failure existed). **The symmetric direction in
this scenario emits noise, not signal.**

### 4.2 Does the symmetric direction have a known-artifact pattern?

bwfr's investigation establishes that the resolver's clobber risk
scales with how *linearized* a schedule is — more
write-after-write-after-read patterns crammed together produce more
artifacts. The default schedule is more linear than CMS by
construction. So the resolver-artifact direction is asymmetric:
**default has more artifact-emission surface than CMS has.**

That said, CMS *can* produce artifacts. Two situations where it
might:

1. **CMS pipelines a write past a same-iteration read it shouldn't
   pass.** This is a real bug, not an artifact, and `OrderInvertedFailure`
   already fires on the missing-side ref edge — covered.

2. **CMS pipelines so a *next-iteration* write becomes the latest
   writer for a same-iteration read.** This would be a real
   inversion (cross-iteration dataflow), and `iter_delta` on the
   missing-side edge surfaces it.

I cannot identify a CMS-side analog of bwfr's "default linearizes
Pack-then-MFMA so the resolver clobbers." Pipelining inherently
*reduces* the clobber risk. So §7.3's symmetric carve-out probably
doesn't need to exist.

### 4.3 uqoz's framing

`SECTION_7_3_SUPPRESSION_DISCUSSION.md` §7.3 (uqoz's memo) explicitly
flags this bead: "If the symbolic-vgpr aliasing produces an *extra*
edge in subj rather than (or in addition to) a missing edge in ref,
the §7.3 gate isn't even the right place to look — `fo40` (the bead
about the symmetric check) is."

The probe shows that scratch-vgpr aliasing in subj *cannot* produce
an extra edge without producing a corresponding missing edge in ref
— the per-byte resolver assigns one producer per read, so the diff
is paired. uqoz's framing was right to flag the question; the
answer is "no" — the symbolic-vgpr-aliasing artifact lives only on
the linearized side, never on the pipelined side, in any setup the
resolver can manufacture.

### 4.4 What the symmetric check would have to absorb

If option 2 or 3 were implemented, the symmetric path would need
its own equivalent of the wx9.3 §6.1 legitimate-CMS-reorder branch
to absorb scenario 2's diff. The check would be:

```python
extra_keys = subj_keys - ref_keys
for key in extra_keys:
    subj_edge = subj_edges_by_key[key]
    # Mirror of legitimate-CMS-reorder: does ref have the same logical
    # (producer.identity, consumer.identity, edge_kind, byte_offset)
    # edge under any (position, operand_slot) tuple?
    if any(e.producer.identity == subj_edge.producer.identity
           and e.consumer.identity == subj_edge.consumer.identity
           and e.edge_kind == subj_edge.edge_kind
           and e.intra_operand_byte_offset == subj_edge.intra_operand_byte_offset
           for e in reference.edges):
        continue  # legitimate ref-mirror reorder
    failures.append(UnexpectedExtraEdge(...))  # genuinely new
```

This is feasible but it duplicates the wx9.3 §6.1 logic. And the
mirror would have to be kept in sync with the original branch
forever. Maintenance liability, not a bug.

---

## 5. Three options reviewed

### Option 1 — Keep asymmetry, document loudly

**What.** Expand `VALIDATOR_DESIGN.md` §4.1 to:
- State the asymmetry as intentional, not provisional.
- Cross-reference this memo for the rationale.
- Document the `validate_edge_wait_coverage` indirect-coverage path
  as the safety net for unintentional CMS extras that are also
  uncovered by waits.
- Cross-reference §7.6 (resolver) for why subj-only artifacts are
  unlikely (pipelining reduces clobber risk).

Add one unit test in `test_dataflow_graph_comparison.py`:
`test_extra_subj_edge_silently_accepted` that pins scenario 1's
behavior — `OrderInvertedFailure` fires once, the extra edge is
silently absorbed. The test asserts what the *current* code does
(no behavior change), making future regressions visible.

**Pros.**
- Zero behavior change. Zero risk of breaking existing tests.
- Documents the design as deliberate.
- Closes the bead with no ongoing maintenance cost.
- Aligns with bwfr's data: the artifact-emission surface is
  asymmetric (default-heavy), so the validator's asymmetric check
  matches the asymmetric problem space.

**Cons.**
- Leaves the small hidden-bug surface from §3.7 unaddressed.
- A future maintainer might re-discover the asymmetry and ask the
  same question. Documentation reduces that cost but doesn't
  eliminate it.

**Effort.** ~1 hour. ~15 lines of doc, ~30 lines of test.

**Risk.** Zero.

### Option 2 — Add symmetric check + new Failure subclass

**What.** Compute `extra_keys = subj_keys - ref_keys`. Mirror the
legitimate-CMS-reorder branch (see §4.4). For unabsorbed extras,
emit a new `UnexpectedExtraEdge(Failure)` subclass.

**Pros.**
- Closes the §3.7 hidden-bug surface.
- Symmetric design — easier to reason about for someone who hasn't
  read this memo.

**Cons.**
- The new Failure is fundamentally weaker than the existing
  Failure subclasses: it can only say "this edge appeared in subj
  and ref doesn't agree it should exist." It cannot classify *why*
  (no `MissingWait`, no `OrderInverted`, no `OverriddenInput` —
  there's no canonical reference to compare *against*).
- Requires building the mirror of wx9.3 §6.1 (~20 lines of new
  code in `compare_graphs`).
- Will fire on scenario-1-shape paired diffs unless suppressed
  when an `OrderInvertedFailure` is already fired for the same
  consumer (which is its own can of worms — failures don't
  currently know about each other).
- New failure subclass needs a `_format_canonical()` method whose
  message is intrinsically vague ("CMS contains an extra edge X→Y;
  the default schedule does not. This may be intentional; review
  required.") — that's a `print` statement, not a typed Failure.
  The validator's "no silent ignores, no vague Failures" contract
  (per VALIDATOR_DESIGN.md §1) would be weakened.
- If real captures *do* have legitimate CMS-side extras (and bwfr
  + the probe suggest they often will, for any realistic kernel
  with reorders), this becomes a noise source that needs its own
  carve-out — recreating the §7.3 problem on the symmetric side.

**Effort.** ~3-5 hours including tests + docs.

**Risk.** Medium. Likely to break existing tests (any test exercising
a real CMS-vs-default capture pair would suddenly see a stream of
new `UnexpectedExtraEdge`s). The breakage shape is "the test corpus
needs to opt-in to expecting these," which is a real-work
multiplier.

### Option 3 — Symmetric check + known-benign list

**What.** Option 2, plus an explicit allowlist of subj-only edge
shapes that should be silently dropped. The allowlist would
include at least:

- Paired diffs where the consumer ALSO has a missing-side edge in
  the same `compare_graphs` invocation (i.e. don't report the
  extra side of any reorder pair).
- Resolver artifacts on the CMS side (TBD — bwfr's investigation
  did not identify any, but the §7.3 carve-out's symmetric
  analog would live here if one were ever found).

**Pros.**
- All of option 2's benefits.
- The "paired-diff drop" rule cleanly absorbs scenario 2's noise.

**Cons.**
- Two carve-out lists (the existing §7.3 + the new symmetric one)
  is more code surface than one.
- The "paired diff" rule effectively means: only report extras
  whose consumer has no missing-edge sibling. After applying that
  rule, what's left? Per §3.6, almost nothing — and what does
  remain is exactly the §3.7 hidden-bug surface, which we cannot
  characterize without a real-capture audit.
- All of option 2's risks.

**Effort.** ~5-8 hours.

**Risk.** Medium-high. The allowlist grows over time; each new
real-capture failure either fits the allowlist (→ allowlist
extension) or is a real bug (→ already covered by waits or
OrderInverted). The pattern of "false-positive surface absorbed
by allowlists" is exactly the §7.3 problem uqoz flagged as
worth re-architecting.

---

## 6. Recommendation

**Adopt option 1.**

Rationale, in priority order:

1. **The probe shows the symmetric direction is mostly noise.**
   Scenario 2 is the dominant CMS-vs-default diff shape, and the
   symmetric check would fire 2 false positives on it that would
   require their own absorption mirror.

2. **bwfr's investigation establishes that the resolver-artifact
   direction is asymmetric.** Default-side captures linearize and
   are vulnerable to clobber; CMS-side captures pipeline and are
   not. The validator's check matches the actual asymmetric
   problem space.

3. **`validate_edge_wait_coverage` already provides a safety net**
   for the §3.7 hidden-bug surface. An extra CMS edge that lacks a
   covering wait WILL fire as `MissingWaitFailure`. The only thing
   the symmetric check would add is "extra edge that IS
   wait-covered but doesn't exist in default" — and bwfr's
   real-capture data does not show this pattern.

4. **The new Failure subclass would be intrinsically weak.** Every
   other Failure carries a typed reason ("missing wait,"
   "insufficient drain," "order inverted"); `UnexpectedExtraEdge`
   would have only the vague "doesn't exist in default." That
   weakens the validator's "knows or admits it doesn't know"
   contract per VALIDATOR_DESIGN.md §1.

5. **Maintenance cost is real.** The mirror-branch + symmetric
   carve-out list would track the wx9.3 §6.1 logic and §7.3 carve-out
   on the symmetric side, doubling the surface that has to be kept
   in sync.

The work proposed for option 1:

- Update `VALIDATOR_DESIGN.md` §4.1 with the rationale (cross-ref
  this memo, bwfr's memo, and §7.6 resolver semantics).
- Update the docstring on `compare_graphs` (`CMSValidator.py:2422-2441`)
  to state the asymmetry as deliberate, with a one-liner pointer
  to §4.1.
- Add a positive unit test
  (`test_dataflow_graph_comparison.py::TestExtraSubjEdgeAccepted`)
  pinning scenario 1's behavior.

**Effort: ~1 hour. Risk: zero.**

---

## 7. Open questions for the user

These are the questions that would change the recommendation if the
answer is unexpected:

1. **Has anyone audited a real-capture CMS-vs-default diff for
   subj-only edges that ARE covered by waits?** §3.7 is the only
   surface I cannot rule out. If you have telemetry from a few
   production builds dumping `subj_keys - ref_keys` and confirming
   it's empty (or only contains scenario-2-shape paired noise),
   that would directly validate option 1. If telemetry shows a
   non-trivial number of unpaired extras, that's evidence for
   option 2 or 3.

2. **Is there appetite for a `validate_edge_wait_coverage`-driven
   "diff-only" mode** that walks `subj_keys - ref_keys` and
   surfaces any edge there that lacks a covering wait? This is a
   cheaper variant of option 2: rather than emit a generic
   `UnexpectedExtraEdge`, run the existing wait-coverage check
   against just the diff. The output would be typed
   `MissingWaitFailure` etc., not a new vague Failure. Effort:
   small. I did not score this as a fourth option in §5 because
   it's structurally the same as option 1 + a debugging knob, but
   if you'd find the diagnostic useful, it's straightforward.

3. **Should the identity collision in scenario 3** (multiple Pack
   instructions with the same rendered string collapsing to a
   single graph node) be tracked as its own bead? It's not a
   compare_graphs symmetry issue, but it surfaced during this
   investigation as a synthetic-fixture surprise. Real captures
   probably don't hit it because real Packs differ in operand
   vgprs, but the deduplication-by-identity pattern at
   `build_dataflow_graph` Phase 1 may merit its own audit.

4. **If the user accepts option 1, what's the right home for the
   pinning test?** I propose
   `test_dataflow_graph_comparison.py::TestAsymmetricByDesign`
   with two methods:
   - `test_extra_subj_edge_silently_accepted_when_paired_with_missing`
     (scenario 1)
   - `test_extra_subj_edges_silently_accepted_when_paired_with_legitimate_reorder`
     (scenario 2, asserting `compare_graphs == []`)
   This co-locates the asymmetry contract with the rest of
   `compare_graphs`'s positive/negative tests.

5. **Is the bead's "Discuss with user before implementing" tag
   satisfied by a memo-only close + a follow-up doc-patch bead?**
   Per the bead's implementation directives, I have not changed
   any code. My recommendation is to close fo40 with this memo,
   then file a tiny follow-up bead for the doc + test patch
   (option 1 work). The memo's purpose was to make the decision
   tractable; the doc patch is independent low-risk work.

---

## 8. Files referenced

- `compare_graphs`: `projects/hipblaslt/tensilelite/Tensile/Components/CMSValidator.py:2422-2505`
- `diagnose_missing_edge`: `projects/hipblaslt/tensilelite/Tensile/Components/CMSValidator.py:2508-2867`
- `DataflowGraph.edge_keys`: `projects/hipblaslt/tensilelite/Tensile/Components/CMSValidator.py:388-426`
- `validate_edge_wait_coverage`: `projects/hipblaslt/tensilelite/Tensile/Components/CMSValidator.py:2869-2913`
- §7.3 carve-out: `projects/hipblaslt/tensilelite/Tensile/Components/CMSValidator.py:2581-2598`
- Legitimate-CMS-reorder branch (wx9.3 §6.1): `projects/hipblaslt/tensilelite/Tensile/Components/CMSValidator.py:2546-2571`
- Identity-coverage check (symmetric): `projects/hipblaslt/tensilelite/Tensile/Components/CMSValidator.py:2442-2483`
- `_is_alu_producer`: `projects/hipblaslt/tensilelite/Tensile/Components/CMSValidator.py:1842-1894`
- VALIDATOR_DESIGN §4.1: `projects/hipblaslt/tensilelite/Tensile/Components/VALIDATOR_DESIGN.md:213-234`
- bwfr memo (resolver-artifact direction): `projects/hipblaslt/tensilelite/Tensile/Components/CROSS_SUBITER_ALU_FP_INVESTIGATION.md`
- uqoz memo (§7.3 design): `projects/hipblaslt/tensilelite/Tensile/Components/SECTION_7_3_SUPPRESSION_DISCUSSION.md`
- Probe (uncommitted): `projects/hipblaslt/tensilelite/Tensile/Tests/scratch/probe_fo40_extra_edges.py`
- Existing comparison tests: `projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_dataflow_graph_comparison.py`

---

*End of memo. No code changes accompany this document.*
