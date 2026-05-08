# Cross-Subiter ALU Producer False-Positive: Root-Cause Investigation

Bead: rocm-libraries-bwfr.
Investigator: bwfr-investigator (Claude, May 2026).
Working tree: `/home/alvasile/rocm-libraries/.worktrees/bwfr-investigation/`.

---

## 1. Summary

The cross-subiter ALU-producer carve-out at
`projects/hipblaslt/tensilelite/Tensile/Components/CMSValidator.py:2533-2546`
is **not** suppressing "legitimate pipelining". It is suppressing a
**defensible-but-misframed** false positive that originates upstream in
the per-byte `latest_writer` resolver in `build_dataflow_graph`
(`CMSValidator.py:1039-1096`).

Root cause in one paragraph: When a Pack instruction in subiter N+1
writes the **same physical vgpr** that a Pack in subiter N already wrote
(scratch-vgpr reuse — the "v133" pattern noted at
`ScheduleCapture.py:1552`), the per-byte resolver overwrites
`latest_writer[v133]` with the subiter-N+1 writer. In the **default**
capture all Packs are emitted before all MFMAs, so when MFMA(subiter=N)
finally appears in stream order, its read of `v133` resolves to the
**subiter-N+1 Pack** as its producer — even though the Pack-N+1 write
hadn't happened yet at the corresponding point in the CMS-pipelined
stream. The default-side graph thus contains an edge
`Pack(subiter=N+1) -> MFMA(subiter=N)` that has no semantic meaning in
the actual program; CMS contains the semantically correct
`Pack(subiter=N) -> MFMA(subiter=N)` instead. `compare_graphs` sees the
default-side edge as missing from CMS and routes it through Phase 1's
order-inversion check, where it would trigger
`OrderInvertedFailure`. The carve-out is the suppression that prevents
that.

**Recommendation: keep the carve-out as-is for now (lowest risk), but
the cleaner long-term fix is at the capture layer** — see §5.

---

## 2. The carve-out

### 2.1 Source

`CMSValidator.py:2521-2553` (full Phase-1 block, with the carve-out at
`:2533-2546`):

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

### 2.2 Gating logic

The carve-out fires only when **all** of the following hold:

1. The reference edge is missing from the subject (`compare_graphs`'s
   contract).
2. Both producer and consumer live in the same body (`p_node.body_label
   == c_node.body_label`) — this is also the gate for OrderInverted
   diagnosis at all.
3. The reference (default) schedule emitted producer **before**
   consumer.
4. The subject (CMS) schedule emitted producer **after** consumer
   (genuine inversion in the captured streams).
5. The producer is an "ALU producer" per `_is_alu_producer`
   (`CMSValidator.py:1842`). This includes every `Pack*`-categorized
   instruction except 4x4 PackMFMAs (the latter are diverted to the
   quad-cycle gap branch above).
6. Producer and consumer live in **different inner-unroll subiters**,
   per `_node_subiter` (`CMSValidator.py:737`).

The legitimate-CMS-reorder branch at `:2494-2519` short-circuits BEFORE
the carve-out can fire when the same logical
(producer.identity, consumer.identity) edge exists in the subject — but
in the scratch-reuse scenario the subject has a DIFFERENT producer
identity feeding the consumer (Pack-N rather than Pack-N+1), so that
short-circuit doesn't trigger.

### 2.3 Comment is partially stale

The comment claims it "mirrors the same-subiter gate
`_classify_edge_coverage` uses in within-graph mode." That gate **no
longer exists**: `_classify_edge_coverage` at `:2880-2886` documents
its removal:

> Phase 1 — same-body order check is no longer needed here: Sub-B's
> per-byte latest-writer resolver only emits edges where producer is
> before consumer in stream order. Within-graph OrderInverted detection
> is therefore impossible by construction; the cross-graph classifier
> in `diagnose_missing_edge` owns OrderInverted detection (with default
> positions for diagnostics).

So the carve-out's "mirrors" reference is to dead code. The carve-out
itself is still load-bearing, but it's the only OrderInverted suppression
in the entire validator.

---

## 3. Test evidence

### 3.1 Direct exercise

There is **no unit test** that names the cross-subiter Pack-after-MFMA
case directly. `grep -rn "cross_subiter" Tests/unit/` returns only
unrelated MFMA acc-chain timing tests. The carve-out is not covered by
any synthetic fixture.

### 3.2 Indirect exercise — real-kernel captures

The carve-out is **load-bearing for production tests** that exercise a
real KernelWriter build of a TF32 4x4 emulation kernel. With the
carve-out neutralized (probe at
`Tests/scratch/run_with_carveout_off.py`, monkey-patches
`_node_subiter` to `lambda n, nmps: 0` for the duration of each
`diagnose_missing_edge` call), the following tests **fail**:

| Test | Failure shape |
|---|---|
| `test_ScheduleCapture.py::TestRealKernelCapture::test_tf32_4x4_tn_capture_shape` | 768 edge differences, all `PackA3[N] -> MFMA` and `PackB3[N] -> MFMA` cross-subiter pairs |
| `test_ScheduleCapture.py::TestRealKernelCapture::test_tf32_4x4_tn_capture_categories` | same |
| `test_ScheduleCapture.py::TestPhase4DefaultCapture::test_default_capture_populated` | same |
| `test_ScheduleCapture.py::TestPhase4DefaultCapture::test_default_and_cms_captures_both_populated` | same |
| `test_ScheduleCapture.py::TestPhase5DefaultTailCapture::test_n_gl_and_n_ll_populated` | same |
| `test_ScheduleCapture.py::TestPhase5DefaultTailCapture::test_no_false_positive_on_clean_cms_kernel` | same |
| `test_ScheduleCapture.py::TestPhase5DefaultTailCapture::test_n_gl_n_ll_state_resets_after_kernel` | same |
| `test_ScheduleCapture.py::TestDataflowGraphIntegration::test_dataflow_gating_passes_on_clean_cms_kernel` | same |
| `test_ScheduleCapture.py::TestDataflowGraphIntegration::test_dataflow_gating_passes_with_MIArchVgpr_true` | same |
| `test_ScheduleCapture.py::TestPgrPlrCaptureMatrixEndToEnd::test_pgr2_snll_false_matches_baseline` | same |

Sample failure messages (from
`KernelWriter.py:5308`, surfaced as
`AssertionError: Dataflow graph comparison failed for kernel
128x128x32: 768 edge difference(s)`):

```
Producer PackA3[6] @ idx=42 is issued after consumer MFMA @ idx=29.
Producer PackB3[0] @ idx=28 is issued after consumer MFMA @ idx=4.
Producer PackB3[1] @ idx=28 is issued after consumer MFMA @ idx=4.
Producer PackA3[18] @ idx=47 is issued after consumer MFMA @ idx=28.
... [768 such lines]
```

Every reported case has the same shape: a `PackA3[N]` (or `PackB3[N]`)
identifier — i.e., the subiter-3 Pack — listed as an "issued after"
producer for an earlier-vmfma-index MFMA. This is exactly the case the
carve-out exists to suppress.

### 3.3 Synthetic minimal reproduction

`Tests/scratch/probe_carveout.py` (do not commit) constructs the
minimal pair with `make_pack(133, ...)` writing to physical vgpr v133
under categories `PackA0` (subiter 0) and `PackA1` (subiter 1), with
two MFMAs that read v133. The probe demonstrates:

- **With distinct vgprs per subiter** (vgpr 100 for PackA0, vgpr 200
  for PackA1) — `compare_graphs` returns `0` failures, no carve-out
  needed. The two graphs form the same edge set.
- **With shared vgpr v133 (scratch reuse)** — the default-side graph
  builds two edges, both with PackA1 as producer:
    - `PackA1 @ pos(li=1,vi=0,si=1)  ->  MFMA0 @ pos(li=1,vi=0,si=2)`
    - `PackA1 @ pos(li=1,vi=0,si=1)  ->  MFMA1 @ pos(li=1,vi=1,si=0)`
  The CMS-side graph builds the semantically correct:
    - `PackA0 @ pos(li=1,vi=0,si=0)  ->  MFMA0 @ pos(li=1,vi=0,si=1)`
    - `PackA1 @ pos(li=1,vi=0,si=2)  ->  MFMA1 @ pos(li=1,vi=1,si=0)`
  `compare_graphs` reports **2 missing edges** in the diff; with the
  carve-out enabled it returns **0 failures** (the cross-subiter ALU
  branch claims both); with the carve-out off, it returns **1
  `OrderInvertedFailure`** for `PackA1 -> MFMA0`.

The probe definitively shows: the default-side graph's
`PackA1 -> MFMA0` edge is an artifact of the per-byte resolver's
"latest writer in stream order" rule overwriting v133's writer when the
default schedule serializes `PackA0, PackA1` before `MFMA0`.

---

## 4. Root cause analysis

### 4.1 The relationship between symbolic and physical vgprs

The carve-out comment talks about "a Pack in subiter N+1 writes a
**symbolic vgpr** that an earlier-subiter MFMA reads under the same
**symbolic name**". This framing is a red herring. The per-byte
resolver at `CMSValidator.py:1039-1096` does not key on symbolic names
— it keys on numeric byte_keys, with symbolic operands resolved to
numeric byte_keys via the body-local `name_to_idx` map
(`CMSValidator.py:1044-1054`,
`ScheduleCapture.py:271-285`). Two operands referring to the same
physical register collapse to a single byte_key entry regardless of
whether either was emitted symbolically.

So the actual mechanism is: **same physical vgpr written by two
different Pack instructions across two different subiters**. The
"symbolic name collision" framing is just one route to that physical
collision; the underlying mechanism is identical for raw-numeric
collisions (probe scenario 2 uses raw numeric `v133` and reproduces the
behavior with no symbolic involvement).

The actual TF32 4x4 production case (judging by the
`PackA3[N]`/`PackB3[N]` identifiers in the failure log and the comment
in `ScheduleCapture.py:1550-1555` referencing "v133 across PackA/PackB
... 24,688 false-positive cross-side OrderInverteds") is per-subiter
Pack scratch reuse — the kernel writer emits all four subiters' worth
of Pack code with the same scratch vgpr range, relying on
write-then-read locality within each subiter. CMS preserves that
locality by pipelining; the default emits them all in linear order,
which destroys the locality at the graph level.

### 4.2 Per-byte resolver behavior

`CMSValidator.py:1042-1096` (Phase 2 of `build_dataflow_graph`) is the
exact site where the artifactual edge originates:

```python
for node in sorted_nodes:                  # ascending stream-position
    ...
    # Phase 2a — reads first
    for read_resource in wrapped.reads:
        for producer, ... in _resolve_producers(
                read_resource, node, latest_writer, ...):
            edges.append(DataflowEdge(producer=producer, consumer=node, ...))
    # Phase 2b — writes second
    for write_resource in wrapped.writes:
        for bk in _byte_keys_for_resource(write_resource, ...):
            latest_writer[bk] = (node, write_resource, w_slot)
```

In the default schedule for the scratch-reuse case, the iteration order
is:

| step | node | action | latest_writer[v133] after |
|---|---|---|---|
| 1 | PackA0 (li=1,vi=0,si=0) | writes v133 | PackA0 |
| 2 | PackA1 (li=1,vi=0,si=1) | writes v133 | **PackA1**  (overwrites) |
| 3 | MFMA0  (li=1,vi=0,si=2) | reads v133 | (resolves to PackA1) |
| 4 | MFMA1  (li=1,vi=1,si=0) | reads v133 | (resolves to PackA1) |

So both MFMAs in the default graph have `PackA1` as their producer for
v133. The intra-subiter edge `PackA0 -> MFMA0` that the kernel writer
intended **does not exist** in the default graph at all.

In the CMS-pipelined schedule:

| step | node | action | latest_writer[v133] after |
|---|---|---|---|
| 1 | PackA0 (li=1,vi=0,si=0) | writes v133 | PackA0 |
| 2 | MFMA0  (li=1,vi=0,si=1) | reads v133 | (resolves to PackA0) |
| 3 | PackA1 (li=1,vi=0,si=2) | writes v133 | PackA1 |
| 4 | MFMA1  (li=1,vi=1,si=0) | reads v133 | (resolves to PackA1) |

CMS produces the semantically correct edges
`PackA0 -> MFMA0` and `PackA1 -> MFMA1`.

The asymmetry between default and subject is **purely a function of
emission order interacting with a destructive last-writer-wins
resolver**, not a function of actual program semantics. The kernel
writer correctly understood that PackA0 produces v133 for MFMA0 and
PackA1 produces v133 for MFMA1; neither schedule has any "real reorder"
of a real dependency. The default capture is just an unfortunate
linearization that aliases dependencies the per-byte resolver cannot
disambiguate.

This is documented (defensively) in `CMSValidator.py:1024-1034`:

> NO subiter scoping. A vgpr is one physical register; whoever wrote it
> most recently in stream order is what every subsequent read sees,
> regardless of which subiter logically "owns" it. If a kernel writer
> mis-pipelines a prefetch (e.g., PackA1 writes v133 before PackA0's
> subiter-0 consumer reads it), the resolver faithfully reports PackA1
> as the producer — the same garbage value the GPU will read.
> compare_graphs then surfaces the divergence. Adding per-subiter
> scoping would HIDE such scheduling bugs to make diagnostics look
> cleaner — the wrong tradeoff.

The author of the resolver explicitly rejected per-subiter scoping
because it would mask real reordering bugs. The carve-out is the
downstream consequence: since the resolver is intentionally aggressive,
the cross-graph classifier has to absorb the artifact at the diagnosis
layer.

### 4.3 Capture-side modeling — is it the actual root cause?

The default-side capture is faithfully recording what the default
scheduler emitted. The default scheduler (SIA / "default-sia3") really
does emit all Packs before all MFMAs linearly within a body, by design.
There is no capture-side "modeling choice" to second-guess — the
capture is structural, not interpretive.

What the **resolver** does on top of that capture is the choice that
manifests the artifact. The resolver's "last writer wins" rule is the
right rule for detecting real reorder bugs (e.g., a CMS that
mis-orders Pack relative to MFMA on the same subiter); but the same
rule is too aggressive when the kernel writer intentionally reuses a
scratch vgpr across subiters and relies on the default scheduler's
linearization being safe-by-locality.

### 4.4 Where in the pipeline the false edge originates

```
ScheduleCapture (faithful)
    └── default schedule emits Packs linearly before MFMAs
           [no problem yet]

build_dataflow_graph Phase 2 / latest_writer resolver
    └── overwrites v133's writer on each Pack
    └── MFMA0's read resolves to Pack-N+1, not Pack-N
           [false edge born here — CMSValidator.py:1083-1096]

compare_graphs
    └── observes the Pack-N+1 -> MFMA-N edge missing from CMS
           [edge difference observed — CMSValidator.py:2433-2452]

diagnose_missing_edge Phase 1
    └── observes default order = Pack before MFMA
        observes subject order = MFMA before Pack
    └── about to emit OrderInvertedFailure
           [classifier would fire here — CMSValidator.py:2532]

The carve-out at :2533-2546
    └── recognizes the cross-subiter ALU-producer signature and suppresses
           [hand-rolled escape valve]
```

The carve-out lives at the bottom of the pipeline. The artifact is
manufactured at `latest_writer` overwrite time, four layers up.

---

## 5. Recommendation

**Keep the carve-out as-is for now.**
**Effort: 0. Risk: 0.** It works, it has clear failure cases (the 10
production-kernel tests above), and removing it would require either
(a) a structural fix in `build_dataflow_graph` or (b) a new and more
discriminating downstream check, both with non-trivial risk surface.

That said, the carve-out is **not the cleanest layer** to fix it at
long-term. Three alternatives, ranked by my confidence in their
correctness:

### Alternative A — physical-write disambiguation in the resolver

When the per-byte resolver overwrites a `latest_writer[bk]` entry,
record the previous writer in a per-key history. When forming an edge
for a read, check if any consumer of the previous writer was emitted
between the previous and current write — if so, the previous writer is
also a (legitimately) live producer of `bk` for some readers and the
current edge is artifactual relative to the live one. This is a real
"may-write" / interval-based dataflow improvement.

- Effort: high (substantially touches `_resolve_producers` and possibly
  the byte_key abstraction).
- Risk: high. The current "single latest writer" rule is precisely what
  makes the resolver's output legible. Going to interval / multi-writer
  semantics is a real model change with broad consequences for every
  edge consumer (`compare_graphs`, `_classify_edge_coverage`, the
  barrier-pattern collector, the wait-coverage simulator).
- Verdict: **do not pursue** as a mitigation for this one carve-out. If
  the resolver model needs to change for OTHER reasons (which I'd
  estimate at 10-20% likelihood as the validator matures), this comes
  along for free.

### Alternative B — scratch-vgpr-reuse detection at the capture layer

`ScheduleCapture` already knows which vgprs are kernel-writer "scratch"
ranges (the writer state distinguishes them from accumulator vgprs).
At capture time, annotate Pack writes to known-scratch vgprs with a
"scratch lifetime ends at next subiter boundary" tag. The resolver
then refuses to publish an edge from a scratch-write in subiter N+1 to
a read in subiter N (the lifetime tag flatly rules it out). This is
the structural fix the validator-author rejected at line 1024-1034 —
explicitly because they wanted the resolver to surface
mis-pipelined-prefetch bugs the GPU would observe as garbage reads.

- Effort: medium (capture-layer change with well-defined boundaries).
- Risk: medium-high. The author's stated concern is real: a
  mis-pipelined prefetch in CMS would also be hidden by this change. A
  complete fix would require differentiating "default-side artifact"
  from "real CMS bug" — but at that point the carve-out is already
  doing exactly that, just at a different layer.
- Verdict: **do not pursue.** This trades the carve-out for a more
  invasive capture-side suppression with the same load-bearing
  responsibility, and additionally weakens the resolver's mis-pipeline
  detection on the CMS side. Strict regression.

### Alternative C — keep carve-out but make it an explicit, documented post-classification filter

Promote the carve-out from an inline branch inside Phase 1 to a named
post-classifier filter (`_filter_default_resolver_artifacts(failures)`)
that consumes the full failure list and drops `OrderInvertedFailure`s
whose `producer.identity` writes a vgpr that is **also written by
another producer in the same body at a different subiter**, AND whose
consumer reads that vgpr from the earlier subiter's write. The check
becomes self-describing in code and can attach an explicit assertion
that the suppression criterion is "default-resolver-aliased
scratch-vgpr reuse, not legitimate pipelining".

- Effort: low (refactor only; semantics unchanged).
- Risk: low. The behavior is identical to today; what changes is the
  comment, the location, and the testability (the filter can be
  unit-tested in isolation against synthetic Failure inputs).
- Verdict: **modest improvement; acceptable when next touching this
  area, but not worth a dedicated change.**

### My top recommendation

**Do nothing this cycle. Annotate the existing carve-out with a
back-pointer to this memo, and treat it as a known artifact of the
intentional resolver design choice at `CMSValidator.py:1024-1034`.**
The user's framing in the bead is partially correct — the carve-out is
"hand-rolled" and looks suspicious — but the framing's claim that it
suppresses an `OrderInvertedFailure` for a real default-emits-A-then-B
pattern under-models what's happening: the default-side edge that
becomes an `OrderInvertedFailure` is itself an artifact of the
default schedule's linearization being a worst case for a per-byte
resolver. The real defect (if any) is the resolver's intentional
aliasing on scratch reuse. The validator author chose to keep the
resolver aggressive (so real CMS bugs get surfaced) and absorb the
default-side artifact downstream — the carve-out is the absorber.

If the user wants to revisit, **Alternative A** is the principled
direction; **Alternative C** is a fine cosmetic improvement.

---

## 6. Open questions for the user

1. **Is the resolver's intentional "no per-subiter scoping" stance still
   the right tradeoff?** The comment at
   `CMSValidator.py:1024-1034` makes the case clearly — surface real
   mis-pipeline bugs, even at the cost of some default-side artifacts.
   This investigation provides one concrete data point for that
   tradeoff but doesn't reopen the decision; if the answer is yes, the
   carve-out is correct as-is.

2. **Are the 10 production-kernel tests in §3.2 considered "real test
   coverage" for this carve-out?** They cover the carve-out via
   end-to-end build, but no synthetic test names it. If you want
   targeted coverage, the probe in `Tests/scratch/probe_carveout.py`
   can be promoted to a permanent fixture in
   `test_dataflow_graph_comparison.py` (a `TestCrossSubiterAluProducer`
   class with two cases: (a) carve-out engaged returns `[]`, (b) when
   the producer/consumer pair is NOT cross-subiter the same shape
   correctly emits `OrderInvertedFailure`). Worth ~30 lines.

3. **Is the related bead `rocm-libraries-uqoz` (referenced in
   `VALIDATOR_DESIGN.md:386-388`) the intended place for any
   refactor work?** If so, a one-line back-reference from the carve-out
   site to that bead would close the loop. (No code change in this
   bead per its constraints.)

---

## 7. Files referenced

- Carve-out: `projects/hipblaslt/tensilelite/Tensile/Components/CMSValidator.py:2521-2553`
- Resolver: `projects/hipblaslt/tensilelite/Tensile/Components/CMSValidator.py:1014-1096`
- `_is_alu_producer`: `projects/hipblaslt/tensilelite/Tensile/Components/CMSValidator.py:1842-1894`
- `_node_subiter`: `projects/hipblaslt/tensilelite/Tensile/Components/CMSValidator.py:737-753`
- `_classify_edge_coverage` (within-graph; carve-out's "mirror" — gone): `projects/hipblaslt/tensilelite/Tensile/Components/CMSValidator.py:2864-2960`
- Pack capture rules / scratch-reuse history: `projects/hipblaslt/tensilelite/Tensile/Components/ScheduleCapture.py:1550-1610`
- `LoopBodyCapture.name_to_idx`: `projects/hipblaslt/tensilelite/Tensile/Components/ScheduleCapture.py:267-287`
- Pack subiter tagging: `projects/hipblaslt/tensilelite/Tensile/Components/ScheduleCapture.py:770-781`
- Design doc note: `projects/hipblaslt/tensilelite/Tensile/Components/VALIDATOR_DESIGN.md:373-388`
- Probe (uncommitted): `projects/hipblaslt/tensilelite/Tests/scratch/probe_carveout.py`
- Pytest harness with carve-out neutralized (uncommitted): `projects/hipblaslt/tensilelite/Tests/scratch/run_with_carveout_off.py`
