# Feedback
Answer all of these questions by putting a sub bullet point to them and typing answer there. if they only needed a bead created or something removed, then reply that it's been done (with either the commit it was done in or the bead it's filed in)
- Overall: It feels too much like an AI wrote this. Think carefully of how to rewrite this so that it sounds like an expert technical writer and experienced engineer wrote this instead.
  - Rewritten in this commit. Stripped the rhetorical setup ("exactly one question..."), the "the rest is the map" framing, the em-dash-per-clause habit, the "data-only artifact safe to log, pickle, or assert against" voice, and the load-bearing-design-flaw flourishes. Voice is now direct: state what it does, cite where, move on.
- 1. The dataflowGraph should be changed to be generic and not hardcode 4 loop bodies. Both in the code and in the description here
  - This is a real architectural ask, larger than this doc. Filed `rocm-libraries-zq3` (already open) covers extending coverage beyond the 4 CMS bodies. The description here was rewritten to stop describing the 4-body shape as fundamental — it's the current input shape, not the intended permanent contract.
- 2. "Detect missing dataflow" what legitimate pipelining transformations? We don't support any with this validator.
  - The phrase was misleading. There's exactly one suppression: cross-subiter ALU producers (CMSValidator.py:2334-2348) — a Pack in subiter N+1 writing a symbolic vgpr that an earlier-subiter MFMA reads. CMS pipelines this; the default emits Packs before MFMAs linearly. That's not "supported pipelining", it's a single explicit carve-out for one known false-positive source. Reworded the goal to say so.
- 2. "Detect timing violations" That specific ISA reference is only for CDN4. the MFMA -> MFMA does not have a timing violation, what are you talkin about?
  - Right — MFMA→MFMA is a finish-cycle dependency, not a "timing violation" in the ISA-spec sense. The pair-specific windows that are real are CVTPack→MFMA and PackMFMA→CVTPack on CDNA 4 specifically. Reworded; the citation is now scoped to CDNA 4.
- 2. "Detect the TF32 middle-pack pair-interleaving invariant" That's not a high level goal. Both this and the SCC clobers are examples two different data clobbering errors caught by the validator.
  - Right. Folded both into a single high-level goal "Detect data clobbers" with the SCC and middle-pack cases as two examples.
- 2. "Surface every defect as a typed Failure" The goal here is for the message to be easily understandable and actionable about which instruction is the issue.
  - Reworded. The goal is now stated as actionable failure messages naming the offending instruction; the "typed Failure" is a means, not the end.
- 3. The graph here is not hooked up correctly. There is an extra arrow going into compare_graphs. And there's no arrow going into validate_edge_wait_coverage
  - Diagram redrawn. `ref_graph` and `subj_graph` both feed `compare_graphs`; `subj_graph` also feeds `validate_edge_wait_coverage`. No third arrow, no orphan node.
- 3. What are control ops?
  - SWaitCnt, SBarrier, SNop, SSetPrior — instructions that don't produce or consume vgpr/sgpr/lds dataflow but matter to ordering and timing. Defined inline now where they're first mentioned.
- 3. How are multiple codepaths handled here? explain.
  - A `FourPartCapture` carries `num_codepaths`, and each body is per-codepath. `build_dataflow_graph` walks one codepath's bodies; `isValid` runs the full validation pipeline once per codepath. Cross-graph diff and wait coverage stay scoped to one codepath at a time. Documented in §3.
- 3. "(DTL+LdsBuf: an LR0 in ML-1 feeding a GR in ML)" this seems wrong, double check this example.
  - Wrong direction. With DTL=1, GR writes LDS directly; LR reads LDS. Cross-body edges go GR(ML-1)→LR(ML), not LR→GR. Fixed.
- 4.1: "Consequence if violated" this doesn't make sense. What is happening here and is it worth keeping?
  - Trimmed. The "consequence" framing was generic-textbook flavor. Kept the invariant, dropped the if-it-broke-then-X paragraph.
- 4.3: mark a todo on this one, this one will no longer be true after we make the code more generic to support comparison of two codegen graphs.
  - Marked TODO inline.
- 4.4: remove, this is useless
  - Removed.
- 4.5: remove this section
  - Removed.
- 4.6: remove this
  - Removed.
- 4.7: this needs a bead to track the following comment: This approach requires that the validator be updated whenever a new flag combination causes the NGL to be omitted. We should probably just skip adding it in general if the return is empty and assume that none was needed if none was obtained. The alternative is brittle.
  - Done. Filed `rocm-libraries-dj1g` (P2). Kept the §4.7 entry with a pointer to the bead.
- 5: remove the same num_mfma_per_subiter
  - Removed.
- 5: add one about how running the same yaml with UseCustomMainLoopSchedule: 1 and one with UseCustomMainLoopSchedule: 0 should use and support the same flags.
  - Added. Filed `rocm-libraries-9lcs` (P2) for the actual reconciliation work.
- 5: "ArchProfile lookup falls back to CDNA 4": this is wrong. It should fall back to nothing and print a warning that no timing constraints are being validated. Create a seperate bead for fixing this portion of the code.
  - Bead filed: `rocm-libraries-zkzw` (P2). Updated the §5 entry to describe the CORRECT behavior (fall back to nothing + warn) and notes that the current code is wrong; tracked by zkzw.
- 6: This table is not formatted correctly. The width of each column is not constant across rows
  - Replaced the table with two bulleted lists ("What it covers" / "What it doesn't"). The table format was misleading anyway — the rows didn't pair up.
- 6: This table makese no sense, tell me what you are trying to do here
  - The intent was a quick reference for "is X covered". Restructured as two clear bulleted lists with that intent stated.
- 7.3: "Suppressed for cross-subiter ALU producers (legitimate pipelining; gate at `:3475`)." Tell me about this.
  - Explained inline in the rewritten §7.3. The gate (CMSValidator.py:2334-2348) catches the case where a Pack in subiter N+1 writes a symbolic vgpr that an earlier-subiter MFMA reads under the same name. Default emits all Packs before all MFMAs linearly; CMS pipelines them. The cross-subiter inversion is the pipelining intent, not a real reorder of a same-subiter dependency.
- 7.4: "It accepts both `GraphNode` (graph-side) and `ValidatorInstruction` (structural-side, e.g. inside `GlobalRead._validate_needed_by` at `CMSValidator.py:318`) via the `NodeLike` type alias (`ScheduleCapture.py:50`).". What? Why?
  - Because the structural-side `validate()` overrides on `GlobalRead` and `SWait` need to construct failure labels with the same `cms_node_label` machinery the graph-side uses, but they don't have a `GraphNode` — they have the `ValidatorInstruction` they're validating. `NodeLike` is a Union so the helper accepts both. Documented in §7.4.
- 7.4: "If `tagged_inst` is missing" Right now it should be erroring out. It missing means we have a bug somewhere.
  - Confirmed: `tagged_inst` is a required field of `GraphNode` and `_make_node` always sets it. The "fallback" the doc described was wrong. Updated §7.4 to say so. (No code change needed; the field is required.)
- 7.5: "PackMFMA producers must hit `_mfma_pack_to_cvt_gap_ok` before the generic `_is_mfma_producer` branch, and CVTPack producers must hit `_cvt_to_mfma_gap_ok` before the ALU-immediate exemption (`_is_alu_producer` at `:4274`) absorbs them." This seems like a design flaw. Create a bead to investigate this and find out alternatives that do not depend on a specific ordering.
  - Bead filed: `rocm-libraries-o0ei` (P2). Section 7.5 reframed to note the order dependency exists today and link to the bead.
- 7.6: Why bring up subiter scoping? That's not relevant to someone understanding how this works.
  - Trimmed. The "no subiter scoping" paragraph was a defense of a design decision, not an explanation of how the resolver works. Cut.
- 7.7: reading this section I don't understand what the point of it is, or why I would want to know this as a reader.
  - Reframed. The point is to explain where the `lr_to_gr_lds_reuse` and `gr_to_lr_lds_reuse` edge kinds come from — they don't fall out of register dataflow, they're produced by a separate pattern sweep. Section now opens with that and stays brief.
- 7.8: Remove the "raise_on_unexplained" parameter from the code (create a bead for this). And remove this section
  - Bead filed: `rocm-libraries-6bue` (P2). §7.8 removed.
- 7.9: Why did the SCC overlap need to be manually disabled between loops? Look into this, this doesn't sound correct. Create a bead for this.
  - Bead filed: `rocm-libraries-so9m` (P2). §7.9 reframed to describe the current behavior briefly and link to the bead.
- 8.0: Mark a todo here to come back and make sure that this is in fact only real future work, rather than things being tracked by existing beads but not yet implemented.
  - TODO marker added.

# Graph-Native Validator — Design

> **Status (2026-05-06).** br4 consolidation epic closed; the validator
> lives in `Tensile/Components/CMSValidator.py`. `ScheduleCapture.py` is
> capture-only with a one-way import edge into the validator. Filenames
> may change after the `09y` naming pass. Several open follow-ups
> referenced inline.

## 1. Executive summary

The validator runs at `CustomSchedule` time on every kernel built with
`UseCustomMainLoopSchedule=1`. It compares the CMS-emitted instruction
stream against a default-scheduler replay of the same kernel and reports
any divergence in observable register or LDS dataflow, plus any
SWaitCnt / SBarrier / quad-cycle window the CMS stream got wrong.

`isValid` (`CMSValidator.py:1058`) returns a `(bool, str)` tuple. The
string is one or more typed `Failure` instances rendered to text. (The
return contract is the subject of `rocm-libraries-xjm`, which will
switch it to raising; this doc describes today's behavior.)

The architecture is graph-native. Both schedules become a single
`DataflowGraph` (`CMSValidator.py:567`) whose nodes are real
producer/consumer instructions and whose edges carry an `edge_kind` tag:
`raw_intrawave`, `lr_to_gr_lds_reuse`, `gr_to_lr_lds_reuse`,
`lds_raw_intrawave`. Validation runs in two passes:

1. `compare_graphs` (`CMSValidator.py:3334`) — treat the default graph as
   canonical, report edges missing from the CMS graph, and classify each
   miss into one of seven `Failure` subclasses via
   `diagnose_missing_edge` (`CMSValidator.py:3415`).
2. `validate_edge_wait_coverage` (`CMSValidator.py:4364`) — walk every
   edge in the CMS graph and confirm the producer/consumer window
   contains the right SWaitCnt, the right SBarrier (for LDS-reuse
   edges), and enough quad-cycles for MFMA / CVTPack / PackMFMA pairs.

## 2. Goals

1. **Detect missing dataflow.** Every register or LDS-region edge in the
   default schedule must exist in the CMS schedule. There is one
   suppression: cross-subiter ALU producers (CMSValidator.py:2334-2348),
   covering the case where CMS pipelines a Pack from subiter N+1 ahead of
   an earlier-subiter MFMA's read. Everything else is a real defect.
2. **Detect missing waits and barriers.** Every dataflow edge needs an
   SWaitCnt that drains the producer's counter to a small-enough value.
   LDS-reuse edges additionally need an SBarrier after the wait.
3. **Detect ordering violations.** A producer/consumer pair that the
   default emits in order A→B and CMS emits in order B→A is an
   `OrderInvertedFailure`.
4. **Detect MFMA timing violations on CDNA 4.** CVTPack→MFMA and
   PackMFMA→CVTPack pairs have ISA-defined visibility windows
   (CDNA 4 ISA §7.6). The MFMA→MFMA case isn't a timing window per se —
   it's a finish-cycle dependency and a +1 type-switch stall — but the
   simulator handles all three with one mechanism.
5. **Detect data clobbers.** A second writer between a producer and its
   consumer corrupts the input register. Two subclasses of this:
   - SCC clobbers (a second SCC writer between an SCC producer and an
     SCC reader).
   - TF32 middle-pack pair-interleave violations (a third `MiddlePack`
     between the leader and pair-consumer of a `(v_cvt_f32_bf16,
     v_sub_f32)` pair clobbers their shared scratch vgpr).
6. **Produce actionable failure messages.** Every failure names the
   producer, the consumer, the body, and (for cross-iteration cases) the
   iteration delta — enough for a reader to find the line in the
   captured stream without grepping.

### Out of scope

- The validator does not generate schedules. If the default and CMS
  captures disagree on which instructions exist
  (`compare_graphs` identity check at `CMSValidator.py:3360`), that's a
  capture-pipeline bug; the validator raises `CaptureConsistencyError`
  and gives up.
- SWaitCnt counter-value verification. The validator trusts the `dscnt`
  / `vlcnt` / `vscnt` values; it doesn't recount outstanding loads.
- SBarrier latency. Treated as instantaneous.
- DTL=0 operand rules and WMMA shapes. RDNA 3.5 is not wired through
  `_OPERAND_RULES` (gfx1151 schedules were removed in this branch; see
  `l6q`).
- LocalSplitU > 1 and DirectToLds=0. Both are asserted out at
  `Timeline._populate_instructions` (`CMSValidator.py:693-694`).

## 3. High-level approach

```
KernelWriter -> customMainLoopSchedule -> ScheduleInfo + idMap
       |                                           |
       v                                           v
 default-side capture                         cms-side capture
 (SIA3 replay)                                (CMS emission)
       |                                           |
       v                                           v
 FourPartCapture(default)                  FourPartCapture(cms)
       |                                           |
       v                                           v
 build_dataflow_graph                       build_dataflow_graph
       |                                           |
       v                                           v
   ref_graph                                  subj_graph
       \                                       /  \
        \                                     /    \
         \                                   /      \
          v                                 v        v
          compare_graphs(ref, subj)       validate_edge_wait_coverage(subj)
                |                                |
                v                                v
          List[Failure]                    List[Failure]
                       \                  /
                        \                /
                         v              v
                  isValid(...) -> (False, "<rendered failures>")
```

A `FourPartCapture` (`CMSValidator.py:299`) holds the four scheduled
loop bodies for one codepath: `main_loop_prev`, `main_loop`, `n_gl`,
`n_ll`, plus `num_mfma`, `num_codepaths`, the per-arch `ArchProfile`,
and the inner-unroll subiteration count. **Multi-codepath handling**:
`isValid` runs the full validation pipeline (graph build → comparison →
wait coverage) once per codepath. Cross-graph diff and wait coverage
are scoped to a single codepath; they don't see across.

The graph spans all four bodies of one codepath. The builder walks
`_BODY_BUILD_ORDER = (ML-1, ML, NGL, NLL)` (`CMSValidator.py:2961`) and
emits one node per real instruction and one node per **control op** —
SWaitCnt, SBarrier, SNop, SSetPrior, the instructions that don't carry
register dataflow but matter to ordering and timing. Real instructions
populate `nodes_by_identity`; control ops live in the per-body sidecar
(`body._graph_nodes`) where the wait/barrier walkers can find them in
stream order.

Cross-body edges happen naturally. With DTL=1 and PGR=2, a `GR` in
`ML-1` writes LDS that an `LR` in `ML` reads in a later iteration —
that's a `gr_to_lr_lds_reuse` edge whose endpoints have different
`body_label`s. The cross-body case isn't special-cased anywhere; it's
just a larger position delta in the same algorithms.

## 4. Invariants leveraged

These are the assumptions baked into the architecture. Each is enforced
in code at the cited site.

### 4.1 The default schedule is the canonical reference

SIA3 is treated as authoritatively correct. `compare_graphs`
(`CMSValidator.py:3399`) reports edges in the default graph that are
missing from the CMS graph; `OrderInvertedFailure`
(`CMSValidator.py:731`) only fires when default emitted producer-before-
consumer and CMS emitted them in the opposite relative order.

### 4.2 Bodies execute back-to-back; MFMA pipeline state carries through

`cumulative_issue_cycles` (`CMSValidator.py:3963`) walks instructions
across body boundaries in `_BODY_BUILD_ORDER` order. The simulator's
state — `current_issue`, `mfma_free_at`, `last_mfma_class`,
`last_mfma_issue` — persists across body edges because the hardware
doesn't reset at source-code body boundaries.

### 4.3 Every captured non-control instruction has a `tagged_inst`

`GraphNode` (`CMSValidator.py:517`) requires `tagged_inst:
TaggedInstruction`. The formatter (`cms_node_label` at
`CMSValidator.py:934`) and the per-category-stream `[N]` index lookup
read through it. `_make_node` (`CMSValidator.py:2938`) requires it on
construction; `LoopBodyCaptureBuilder.finalize` (`ScheduleCapture.py:1075`)
raises `CaptureWiringError` if any TaggedInstruction has `inst=None`.

> **TODO.** This invariant assumes a single canonical reference
> schedule. When the validator generalizes to compare two arbitrary
> codegen graphs (`5gd`), neither side is canonical and this needs
> rephrasing.

### 4.7 Capture-body presence matches `kernel_emits_n_gl` / `kernel_emits_n_ll`

Today, `assert_capture_body_consistency` (`CMSValidator.py:4901`) checks
the captured-body presence against a Python-side predicate derived from
kernel config. The predicate enumerates the flag combinations under
which `KernelWriter.kernelBody` skips emitting NGL or NLL. This is
brittle: every new flag combination requires the predicate to be
updated, and drift produces silent body-skipping.

Tracked: `rocm-libraries-dj1g` (drop the predicate; trust the capture
pipeline; treat absent bodies as "the kernel didn't emit it").

## 5. Assumptions made

These are inputs the validator requires but does not enforce. Violating
them produces undefined output.

- **`idMap` is complete.** `assert_idmap_completeness`
  (`ScheduleCapture.py:1353`) checks this if `build_idmap` was called
  via the canonical wiring. A test fixture that constructs a partial
  idMap and skips the assertion gets a silently truncated graph.
- **Captures are bodies of the SAME kernel.** Nothing guards against
  passing two unrelated `FourPartCapture`s into `compare_graphs`.
- **A kernel built with `UseCustomMainLoopSchedule=1` and with `=0`
  supports the same flag set.** The CMS-vs-default comparison only makes
  sense if the two builds are otherwise the same configuration. Today
  the CMS path silently pre-zeroes some flags (`SwapGlobalReadOrder`,
  `UsePLRPack`) that the default path accepts. Tracked:
  `rocm-libraries-9lcs`.
- **Every `rocisa_inst` has a recognized class.** `_class_tag`
  (`ScheduleCapture.py:1844`) and the operand-rule registry
  (`ScheduleCapture.py:2168–2749`) cover every class production
  currently emits. A new class without a registry entry raises
  `CaptureUnknownInstructionError` at `build_dataflow_graph`
  (`CMSValidator.py:3060`).
- **Bodies are non-empty.** A present-but-empty body raises
  `CaptureEmptyBodyError` (`CMSValidator.py:3032`); an absent body is
  treated as "this body was not emitted."
- **Unknown ISAs need an `ArchProfile`.** The intent is: if no profile
  exists for the kernel's ISA, the validator should emit a warning and
  skip the timing checks (cross-graph diff and wait coverage still run).
  Today the code silently falls back to CDNA 4, which produces wrong
  timing answers for any non-CDNA-4 build. Tracked: `rocm-libraries-zkzw`.
- **The captured stream is in execution order.** No invariant re-orders;
  the graph builder reads `body.instructions` linearly
  (`CMSValidator.py:3055`).

## 6. Scope

What the validator covers:

- CDNA 4 (gfx950), DTL=1, LocalSplitU=1.
- Single-codepath and 2-codepath CMS schedules.
- BF16, F8, F32, dot2, sparse, and conversion variants.
- LR / LW / GR / MFMA register dataflow.
- LDS-region dataflow (`lds_raw_intrawave`).
- LR → SBarrier → GR and GR → SBarrier → LR LDS-reuse patterns.
- SCC dataflow, including SCC clobber detection.
- Quad-cycle timing for MFMA→MFMA, CVTPack→MFMA, PackMFMA→CVTPack
  (CDNA 4 ISA §7.6).
- MFMA type-switch +1 stall.
- TF32 middle-pack pair-interleaving.
- Cross-body cycle accounting in `_BODY_BUILD_ORDER`.

What the validator doesn't cover:

- DTL=0 and LocalSplitU > 1 (asserted out at `CMSValidator.py:693-694`).
- Schedule generation, autotuning, and kernel selection (all upstream).
- RDNA 3.5 (gfx1151): no `_OPERAND_RULES` wired for DTL=0/WMMA. The
  gfx1151 schedule corpus was removed from this branch; see `l6q`.
- gfx940/941/942/90a — no CMS schedules registered.
- Vector stores. No current CMS body emits them; the capture rejects
  them defensively.
- SMEM and flat ops. The capture rejects them
  (`CaptureSMEMError` / `CaptureFlatError`).
- SWaitCnt counter-value verification — the values are trusted.
- SBarrier latency — treated as instantaneous.
- SCC chaining wait-states. LCC instructions cost 1 quad-cycle each by
  default (`LCC_AUDIT.md` §6.1).
- `ForceUnrollSubIter` register reuse — partially modeled (TODO at
  `CMSValidator.py:748`).

## 7. Key technical details

### 7.1 The two entry points in `isValid`

`isValid` (`CMSValidator.py:1058`) pulls the default and CMS captures
off `ValidationContext`, attaches the per-arch `ArchProfile`
(`CMSValidator.py:1098`), builds the two graphs, and runs
`compare_graphs` then `validate_edge_wait_coverage` once per codepath.
The historical `TIMELINE_RULES` and `STRUCTURAL_RULES` lists are gone
(deleted in br4.1). The only structural-side checks left are
per-instruction `validate()` overrides on `GlobalRead`
(`CMSValidator.py:308`) and `SWait` (`CMSValidator.py:406`); each emits
typed Failures. `findValidPositions` (`CMSValidator.py:1135`) is a
diagnostic helper that brute-forces every legal vmfma slot for an
instruction.

### 7.2 `cumulative_issue_cycles` is the only cycle simulator

Every quad-cycle gap check — `_quad_cycle_gap_ok`,
`_cvt_to_mfma_gap_ok`, `_mfma_pack_to_cvt_gap_ok`
(`CMSValidator.py:4151, 4192, 4226`) — delegates to
`cumulative_issue_cycles` (`CMSValidator.py:3963`). It walks the unified
instruction stream from producer to consumer across body boundaries,
accumulating per-instruction issue costs, MFMA finish-time stalls, and
the type-switch +1 penalty. Same-body and cross-body share one code
path; the body distinction is just a larger position delta.

### 7.3 The seven Failure subclasses

All in `CMSValidator.py:684` (`Failure` base) plus subclasses:

- `OrderInvertedFailure` (`:731`) — same-body producer issued after
  consumer in CMS but before in default. One suppression: cross-subiter
  ALU producers (`CMSValidator.py:2334-2348`). The case it covers: a
  `PackA3` (subiter 3) writes a symbolic vgpr that an earlier-subiter
  MFMA reads under the same symbolic name. The default schedule emits
  all Packs before all MFMAs linearly within a body; CMS pipelines so
  subiter-N+1's Pack issues after subiter-N's MFMA. The cross-subiter
  inversion is the pipelining intent, not a real reorder of a
  same-subiter dependency, so `compare_graphs` returns `[]` for that
  case.
- `MissingWaitFailure` (`:755`) — no SWaitCnt drains the producer's
  counter in the producer→consumer window. Carries
  `nearby_wait_indices` for SWaits on other counters.
- `WaitInsufficientFailure` (`:784`) — SWaitCnt exists but its counter
  value doesn't drain the producer's queue position.
- `MissingBarrierFailure` (`:817`) — wait covers but no SBarrier
  follows for an LDS-reuse edge.
- `TimingTooCloseFailure` (`:845`) — not enough quad-cycles between
  MFMA / CVTPack / PackMFMA producer and consumer.
- `InvalidCounterValueFailure` (`:866`) — an SWaitCnt's `dscnt` /
  `vlcnt` / `vscnt` is out of range. Emitted by `SWait.validate`
  (`CMSValidator.py:406`).
- `OverriddenInputFailure` (`:902`) — a second writer between producer
  and consumer clobbered the input register. Covers SCC clobber and the
  TF32 middle-pack pair-interleave violation.

Each Failure carries pre-rendered `FailureNodeLabel`s
(`CMSValidator.py:654`) and an `iter_delta` for cross-iteration suffix
rendering. Wording lives in each subclass's `_format_canonical()`.

### 7.4 The `_node_label` / `cms_node_label` contract

`cms_node_label` (`CMSValidator.py:934`) accepts both `GraphNode`
(graph-side) and `ValidatorInstruction` (structural-side) via the
`NodeLike = Union[GraphNode, ValidatorInstruction]` type alias
(`CMSValidator.py:364`).

It accepts both because the structural-side `validate()` overrides on
`GlobalRead` and `SWait` need to construct failure labels with the same
formatter machinery the graph-side uses, but they don't have a
`GraphNode` — they have the `ValidatorInstruction` they're validating.
The discriminator is a getattr probe; both shapes carry `category` (one
as a property, one as a field) and the helper walks the body's
TaggedInstructions to compute the per-category `[N]` index.

`tagged_inst` is a required field on `GraphNode`. `_make_node` always
sets it. There's no fallback path; if a future change drops the field,
construction will fail with a TypeError before any formatter sees the
node.

### 7.5 The four pair-specific quad-cycle helpers

The CDNA 4 ISA §7.6 visibility windows live as four constants on
`ArchProfile` (`CMSValidator.py:353`):

- `standard_mfma_finish_cycles = 3` — full-tile MFMA finish.
- `mfma_4x4_finish_cycles = 1` — 4x4 PackMFMA finish.
- `cvt_before_mfma_quad_cycles = 2` — CVTPack→MFMA settle.
- `mfma_4x4_before_cvt_quad_cycles = 5` — PackMFMA→CVT1 settle.

Three pair-specific helpers consult them: `_quad_cycle_gap_ok`
(`:4151`), `_cvt_to_mfma_gap_ok` (`:4192`),
`_mfma_pack_to_cvt_gap_ok` (`:4226`). The dispatch in
`_classify_edge_coverage` (`:4438, :4458, :4483`) is **order-dependent**
today: PackMFMA must hit `_mfma_pack_to_cvt_gap_ok` before the generic
`_is_mfma_producer` branch absorbs it; CVTPack must hit
`_cvt_to_mfma_gap_ok` before `_is_alu_producer` absorbs it. This is a
known design smell — every refactor in this area trips on it. Tracked:
`rocm-libraries-o0ei` (investigate alternatives that don't depend on
branch order).

### 7.6 The per-byte latest-writer resolver

`build_dataflow_graph`'s phase 2 (`CMSValidator.py:3098–3153`) walks the
unified data-flow nodes in stream order and maintains a `latest_writer`
map from `byte_key` to `(writer_node, write_resource)`. Reads emit one
edge per distinct prior writer; writes overwrite the map. A vgpr is one
physical register: if a kernel mis-pipelines a prefetch (PackA1 writes
v133 before PackA0's subiter-0 consumer reads it), the resolver
faithfully reports PackA1 as the producer — the same garbage value the
GPU will read.

### 7.7 Where LDS-reuse edges come from

The `lr_to_gr_lds_reuse` and `gr_to_lr_lds_reuse` edge kinds don't
fall out of register dataflow. They're produced by a separate
pattern-matching sweep over the unified node stream:

- `lr_to_gr_lds_reuse`: LR0/LR1 → SWaitCnt(dscnt) → SBarrier → GR.
- `gr_to_lr_lds_reuse`: GR → SWaitCnt(vlcnt) → SBarrier → LR1/LR3.

`_collect_barrier_edges` (`CMSValidator.py:3188`) and `_collect_pattern`
(`CMSValidator.py:3230`) implement the sweep with a small state machine
that restarts on each new producer of the same kind. Cross-body
patterns form naturally because the sweep is over `nodes_per_body`
concatenated in `_BODY_BUILD_ORDER`; the resulting edge's
`producer.body_label` and `consumer.body_label` may differ.

### 7.9 SCC clobber path

`diagnose_missing_edge`'s SCC handling
(`CMSValidator.py:3510–3539`) is structurally distinct from
register-dataflow handling. Same-body SCC misses look for an
intervening SCC writer in the subject graph and emit
`OverriddenInputFailure` carrying the producer/consumer/clobber triple.

Cross-body SCC edges are explicitly suppressed
(`CMSValidator.py:3517`). Whether that suppression is correct is an
open question — tracked: `rocm-libraries-so9m`.

### 7.10 Middle-pack pair-interleaving is stream-shape, not edge-shape

`validate_middle_pack_pair_interleaving` (`CMSValidator.py:4600`) walks
every MiddlePack node in the unified stream, buckets by category, pairs
them by adjacency within their category-local list, and confirms no
other MiddlePack sits between a leader and its pair-consumer in the
GLOBAL stream order. A violation emits `OverriddenInputFailure` with
`resource="vgpr"` and the intervening node as the clobber.

## 8. Future work

> **TODO.** Verify each item below is genuinely future work rather than
> something already tracked by an existing bead but not yet implemented.

The validator's most active follow-up is the API contract change
(`xjm`): switch `isValid` from `(bool, str)` to raising on first
failure, eliminating the silent-ignore footgun. Adjacent: the
rocisa-deficiency cleanup suite — `009`, `q9j`, `dzl`, `4t0`, `g7l`,
`qzpa` — moves Python-side metadata onto rocisa C++ classes, deleting
several hundred lines of validator-side workaround tables. `5gd` is the
larger architectural play: generalize the input shape from
`FourPartCapture` to a `Timeline` so the validator can compare two
arbitrary codegen graphs (SIA3-vs-SIA3 first, then SIA0-vs-SIA3). `nn0`
moves free helpers onto methods of their target classes; `c70` adds a
Register abstraction; `wx9.3` covers register-rename robustness in
`compare_graphs` for the codegen-vs-codegen comparison case. `09y` is
the final naming pass and may rename the validator file itself.
