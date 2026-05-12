# Feedback
Answer all of these questions by putting a sub bullet point to them and typing answer there. if they only needed a bead created or something removed, then reply that it's been done (with either the commit it was done in or the bead it's filed in)
- Overall: Don't use ML or ML-1 to describe loops, just say ML iter i and i+1. Keep using NGL and NLL when relevant (PGR > 0 and PLR > 0). Done in this commit.
0 overall: there are still references to FourPartCapture, remove them, those are implementation details we don't care about. AND I told you to move on to more generic wording. Done in this commit.
- 2. "Detect missing dataflow" what legitimate pipelining transformations? We don't support any with this validator.
  - The phrase was misleading. There's exactly one suppression: cross-subiter ALU producers (CMSValidator.py:2334-2348) — a Pack in subiter N+1 writing a symbolic vgpr that an earlier-subiter MFMA reads. CMS pipelines this; the default emits Packs before MFMAs linearly. That's not "supported pipelining", it's a single explicit carve-out for one known false-positive source. Reworded the goal to say so.
    - Why is there this false positive at all? File a bead for this to investigate and discuss it with me before taking any action. Done — filed as `rocm-libraries-bwfr`.
- 3. How are multiple codepaths handled here? explain.
  - A `FourPartCapture` carries `num_codepaths`, and each body is per-codepath. `build_dataflow_graph` walks one codepath's bodies; `isValid` runs the full validation pipeline once per codepath. Cross-graph diff and wait coverage stay scoped to one codepath at a time. Documented in §3.
    - This isn't clear enough in the documentation. Done in this commit. (Also corrected the prior reply: today `isValid` only validates codepath 0; multi-codepath iteration is open work, called out explicitly in §3.)
- 4.1: "Consequence if violated" this doesn't make sense. What is happening here and is it worth keeping?
  - Trimmed. The "consequence" framing was generic-textbook flavor. Kept the invariant, dropped the if-it-broke-then-X paragraph.
- 4.1: "reports edges in the default graph that are missing from the CMS graph" What about graphs in the CMS but not in the default?
  - Investigated. `compare_graphs` (`CMSValidator.py:2233`) computes `missing_keys = ref_keys - subj_keys` only — the reverse direction is NOT checked. The identity-coverage check at the top is symmetric, but the edge-set comparison itself is not. An extra CMS edge can still surface indirectly through `validate_edge_wait_coverage` (which walks every CMS edge and demands a covering wait), but the structural fact "CMS introduced a dataflow edge the canonical schedule never had" is not directly reported. Reflected in §4.1. Done — filed as `rocm-libraries-fo40` to discuss whether to add the symmetric check.
- 7.3: "Suppressed for cross-subiter ALU producers (legitimate pipelining; gate at `:3475`)." Tell me about this.
  - Explained inline in the rewritten §7.3. The gate (CMSValidator.py:2334-2348) catches the case where a Pack in subiter N+1 writes a symbolic vgpr that an earlier-subiter MFMA reads under the same name. Default emits all Packs before all MFMAs linearly; CMS pipelines them. The cross-subiter inversion is the pipelining intent, not a real reorder of a same-subiter dependency.
    - Create a bead to discuss this issue with me in more detail. Done — filed as `rocm-libraries-uqoz`.
- 7.4: "It accepts both `GraphNode` (graph-side) and `ValidatorInstruction` (structural-side, e.g. inside `GlobalRead._validate_needed_by` at `CMSValidator.py:318`) via the `NodeLike` type alias (`ScheduleCapture.py:50`).". What? Why?
  - Because the structural-side `validate()` overrides on `GlobalRead` and `SWait` need to construct failure labels with the same `cms_node_label` machinery the graph-side uses, but they don't have a `GraphNode` — they have the `ValidatorInstruction` they're validating. `NodeLike` is a Union so the helper accepts both. Documented in §7.4.
    - Create a bead to discuss this issue with me in more detail. Done — filed as `rocm-libraries-x4ef`.

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
   suppression: cross-subiter ALU producers (CMSValidator.py:2361-2378),
   covering the case where CMS pipelines a Pack from subiter N+1 ahead of
   an earlier-subiter MFMA's read. Everything else is a real defect. The
   root cause of this false-positive is open for investigation —
   tracked: `rocm-libraries-bwfr`.
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
 default kernel capture                     CMS kernel capture
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

The capture for one schedule (default or CMS) is a per-codepath bundle
of the four scheduled loop bodies: `main_loop_prev` (the prior main-loop
iteration retained for cross-iter reasoning), `main_loop`, `n_gl`
(no-global-load loop), and `n_ll` (no-local-load loop). The bundle also
carries `num_mfma`, `num_codepaths`, the per-arch `ArchProfile`, and the
inner-unroll subiteration count.

**What a "codepath" is here.** CMS can emit more than one main-loop
schedule for a single kernel; each one is a "codepath". Different
codepaths execute in different SIMDs and use different scheduling
orders for the same instruction set, but all converge to identical
architectural state by the end of the main-loop iteration. The
codepath count is set at CMS dispatch (`CustomSchedule.py:527` passes
`numCodePath` into `build_cms_four_part_capture`); the default-side
capture is always single-codepath (`{0: body}`), and the CMS tail
loops (`n_gl`, `n_ll`) are also single-codepath because
`_emitNoLoadLoopBodyCMSMacro` hard-codes `\\ID=0` for them — codepath
divergence only exists in `main_loop` / `main_loop_prev`. So
`num_codepaths` matters only for the main-loop bodies.

**How the validation pipeline scopes per codepath.** Each call to
`build_dataflow_graph` (`CMSValidator.py:851`) reads body `0` from each
of the four per-codepath dicts (`by_cp[0]`) and produces one
`DataflowGraph` for that codepath. `compare_graphs` and
`validate_edge_wait_coverage` then operate on that single graph;
cross-graph diff and wait coverage stay scoped to one codepath at a
time and never see across. Today `isValid`
(`CMSValidator.py:3867-3946`) only invokes `build_dataflow_graph` on
codepath 0 — it does not iterate `range(num_codepaths)`. Multi-codepath
iteration of the comparison/wait-coverage passes is open work and
should be tracked separately if it is not already; the CMS-side capture
contains all codepaths' bodies, but the comparison consumes only
codepath 0.

```
kernel capture (one bundle per schedule)
+----------------------+
| main_loop:      {0: body_cp0, 1: body_cp1, ...}   <- per-codepath
| main_loop_prev: {0: body_cp0, 1: body_cp1, ...}   <- per-codepath
| n_gl:           {0: body}                         <- always cp 0 only
| n_ll:           {0: body}                         <- always cp 0 only
| num_codepaths:  N
+----------------------+
       |
       | build_dataflow_graph reads by_cp[0] from each body
       v
   one DataflowGraph per build_dataflow_graph call
       |
       | compare_graphs(ref_cp0, subj_cp0)
       | validate_edge_wait_coverage(subj_cp0)
       v
   per-codepath Failures (today: codepath 0 only)
```

The graph spans all four bodies of the chosen codepath. The builder
walks `_BODY_BUILD_ORDER = (ML iter i, ML iter i+1, NGL, NLL)`
(`CMSValidator.py:2961`; `ML iter i` is the prior main-loop iteration,
`ML iter i+1` is the current one) and emits one node per real
instruction and one node per **control op** — SWaitCnt, SBarrier,
SNop, SSetPrior, the instructions that don't carry register dataflow
but matter to ordering and timing. Real instructions populate
`nodes_by_identity`; control ops live in the per-body sidecar
(`body._graph_nodes`) where the wait/barrier walkers can find them in
stream order.

Cross-body edges happen naturally. With DTL=1 and PGR=2, a `GR` in
`ML iter i` writes LDS that an `LR` in `ML iter i+1` reads — that's a
`gr_to_lr_lds_reuse` edge whose endpoints have different `body_label`s.
The cross-body case isn't special-cased anywhere; it's just a larger
position delta in the same algorithms.

## 4. Invariants leveraged

These are the assumptions baked into the architecture. Each is enforced
in code at the cited site.

### 4.1 The default schedule is the canonical reference

> **NOTE 2026-05-12 — architecture under revision.** The user has
> rejected the shadow-as-reference contract this section describes.
> The "default graph" today is the SIA3 shadow capture
> (`_last_default_capture`), which is not assembled into runnable code
> and never executes. See `2LZD_INVESTIGATION.md §6` for the decision
> and the live approach set ({A, D, H}). Whichever approach lands,
> this section's "default schedule is canonical" framing will need
> revision: under A the canonical reference becomes a real non-CMS
> kernel build; under D the comparison contract is dropped in favor
> of a slot-table check; under H both sides are real CMS builds with
> no canonical/non-canonical distinction.

SIA3 is treated as authoritatively correct. `compare_graphs`
(`CMSValidator.py:2233`) computes `missing_keys = ref_keys - subj_keys`
and reports edges in the default graph that are missing from the CMS
graph; `OrderInvertedFailure` (`CMSValidator.py:731`) only fires when
default emitted producer-before-consumer and CMS emitted them in the
opposite relative order.

The edge-set comparison is **one-way**: the reverse direction
(`subj_keys - ref_keys` — edges in the CMS graph that are absent from
the default graph) is not checked. The identity-coverage check at the
top of `compare_graphs` IS symmetric (it raises
`CaptureConsistencyError` if either side has a data-flow node identity
the other lacks), but the edge-set comparison itself is not. In
practice, an extra CMS edge can still be caught indirectly by
`validate_edge_wait_coverage`, which walks every edge in the CMS graph
and demands a covering SWaitCnt — so an unintended CMS edge that
happens to lack a wait would surface as a `MissingWaitFailure`. But the
structural fact "CMS introduced a dataflow edge the canonical schedule
never had" is not directly reported. Whether to add a symmetric check
is open — tracked: `rocm-libraries-fo40`.

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

### 4.7 Capture-body presence — single source of truth (resolved)

Resolved by `rocm-libraries-dj1g`. The Python-side predicates that
re-derived NGL/NLL emission from kernel config and the consistency
assertion that cross-checked captured-body presence against them were
deleted.

Today: the default-side capture pipeline observes whether the kernel
actually emitted each body. The CMS-side `build_cms_four_part_capture`
takes the default-side capture as a parameter and mirrors its body
shape by construction, so the two captures' n_gl/n_ll presence sets
are guaranteed equal at the point `compare_graphs` runs — no
verification needed.

The expansion is deferred from `customMainLoopSchedule` (which runs
before the default-side capture exists) to `KernelWriter.kernelBody`
(which runs after), with the inputs stashed on the writer as
`_pending_cms_capture_inputs` (a `CmsCaptureInputs` dataclass) in
between.

## 5. Assumptions made

These are inputs the validator requires but does not enforce. Violating
them produces undefined output.

- **`idMap` is complete.** `assert_idmap_completeness`
  (`ScheduleCapture.py:1353`) checks this if `build_idmap` was called
  via the canonical wiring. A test fixture that constructs a partial
  idMap and skips the assertion gets a silently truncated graph.
- **Captures are bodies of the SAME kernel.** Nothing guards against
  passing two unrelated kernel captures into `compare_graphs`.
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
- Quad-cycle timing for CVTPack→MFMA, PackMFMA→CVTPack
  (CDNA 4 ISA §7.6).
- Some MFMA type-switch stall.

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
  ALU producers (`CMSValidator.py:2361-2378`). The case it covers: a
  `PackA3` (subiter 3) writes a symbolic vgpr that an earlier-subiter
  MFMA reads under the same symbolic name. The default schedule emits
  all Packs before all MFMAs linearly within a body; CMS pipelines so
  subiter-N+1's Pack issues after subiter-N's MFMA. The cross-subiter
  inversion is the pipelining intent, not a real reorder of a
  same-subiter dependency, so `compare_graphs` returns `[]` for that
  case. The suppression is open for further discussion — tracked:
  `rocm-libraries-uqoz` (and see `rocm-libraries-bwfr` for the related
  root-cause investigation).
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

### 7.4 The `cms_node_label` contract

`cms_node_label` accepts a single `GraphNode` shape and constructs the
per-node `FailureNodeLabel` carried by every typed `Failure`. It walks
the body's `TaggedInstruction`s to compute the per-category `[N]` index
(plain `MFMA` omits the `[N]`); when `body_capture` is None or the node
has no entry in that body's tagged_inst stream, the helper falls back to
a bare `category` primary.

The historic `NodeLike = Union[GraphNode, ValidatorInstruction]` alias
and the structural-side caller path it served (`GlobalRead.validate` ->
`_validate_needed_by` over a `Timeline`) were deleted in
`rocm-libraries-wa57`. The dead-code rationale and the original
investigation are recorded in
`Tensile/Components/NODELIKE_UNION_DISCUSSION.md`. If a future use case
genuinely needs structural-side label construction, re-add via a
`to_node_label()` method on the new instruction shape with proper
body-context plumbing — git history preserves what was here.


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

- `lr_to_gr_lds_reuse`: LR → SWaitCnt(dscnt) → SBarrier → GR.
- `gr_to_lr_lds_reuse`: GR → SWaitCnt(vlcnt) → SBarrier → LR.

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

### 7.11 VOPD pair-formation (RDNA3.5 §7.6 R-4..R-7) — correctness pass

`validate_vopd_pair_formation` is a **correctness** pass, not a
performance / soft-fail pass. RDNA3.5 §7.6 defines the dual-issue VOPD
encoding and four pair-formation hard rules (the ISA wording: "These
are hard rules — the instruction does not function if these rules are
broken"):

- **R-4** Source bank conflict: SRCX0/SRCY0 must use different VGPR
  banks (banks indexed by `SRC[1:0]`); same constraint for VSRCX1/VSRCY1.
- **R-5** Destination parity: one vdst must be even, the other odd
  (vdstY's LSB is forced to `!vdstX[0]` in the encoding).
- **R-6** SRC2 even/odd: when both X and Y use SRC2 (FMAMK_F32,
  DOT2ACC_F32_F16, DOT2ACC_F32_BF16, FMAC_F32), one SRC2 must be even
  and the other odd.
- **R-7** Independence: X and Y must be independent — no RAW between
  them (VOPD reads the OLD value if both touch the same VGPR), no WAW.

Every violation emits a `VopdPairFormationFailure` carrying
`(rule, instruction_a, instruction_b, why)`. Unlike
`TimingTooCloseFailure`, there is no soft-fail bucket: a §7.6 hard-rule
violation makes the GPU silently produce wrong results, so the
`validate_schedule_against_default` wrapper raises on any
`vopd_failures` list, alongside the hard graph-comparison failures.
R-8 (wave32-only) is a kernel-wide property and is checked at
kernel-config time rather than per-pair, so it is excluded from this
pass.

VOPD pair recognition is via `DataflowGraph.vopd_pairs`, a
`list[VopdPair]` with sources `{src0, src1, src2}` and destination
`vdst` populated for both X and Y operands. SRC0 and SRC2 fields use
`-1` to denote "not a VGPR" (SGPR / inline / literal for SRC0;
"operation does not consume SRC2" for SRC2). The kernel emitter does
not produce VOPD today; `vopd_pairs` is empty for every kernel, the
pass returns `[]`, and the wrapper observes no failures. The pass is
installed unconditionally so that the moment VOPD emission lands, the
gating correctness check is already wired into the validator entry
point (`cms_from_default.validate_schedule_against_default`).

## 8. Future work

> **TODO.** Verify each item below is genuinely future work rather than
> something already tracked by an existing bead but not yet implemented.

The validator's most active follow-up is the API contract change
(`xjm`): switch `isValid` from `(bool, str)` to raising on first
failure, eliminating the silent-ignore footgun. Adjacent: the
rocisa-deficiency cleanup suite — `009`, `q9j`, `dzl`, `4t0`, `g7l`,
`qzpa` — moves Python-side metadata onto rocisa C++ classes, deleting
several hundred lines of validator-side workaround tables. `5gd` is the
larger architectural play: generalize the input shape from the current
per-codepath kernel-capture bundle to a `Timeline` so the validator can
compare two arbitrary codegen graphs (SIA3-vs-SIA3 first, then
SIA0-vs-SIA3). `nn0`
moves free helpers onto methods of their target classes; `c70` adds a
Register abstraction; `wx9.3` covers register-rename robustness in
`compare_graphs` for the codegen-vs-codegen comparison case. `09y` is
the final naming pass and may rename the validator file itself.
