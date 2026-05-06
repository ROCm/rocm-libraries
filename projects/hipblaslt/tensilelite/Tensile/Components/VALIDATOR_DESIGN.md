# Graph-Native Validator — Design

> **Status as of 2026-05-06 (interim).** This doc reflects the validator at
> commit `18cbf1e59b35`. The `br4` consolidation epic is in flight: sub-beads
> br4.1 and br4.2 just landed; br4.3..br4.10 are pending. Every symbol cited
> below by `ScheduleCapture.py:NNNN` is scheduled to migrate into
> `CMSValidator.py` over those sub-beads (see the consolidation audit at
> `Tensile/Components/CMS_VALIDATOR_CONSOLIDATION_AUDIT.md`). Filenames may
> also change after the naming pass (bead `09y`). A final pass will revise
> this doc once all open refactor epics close.

## 1. Executive summary

The validator answers exactly one question for every kernel built with a
Custom Main-Loop Schedule (CMS): *"would this hand-written CMS schedule
produce the same observable register dataflow as the default SIA3 scheduler,
and does the CMS stream actually contain the SWaitCnt / SBarrier instructions
needed to make those edges safe?"*

It runs at `CustomSchedule` time, after CMS has emitted its final scheduled
stream and the default scheduler has been replayed in parallel for the same
kernel. The answer is a `(bool, str)` tuple returned from
`isValid` (`CMSValidator.py:1058`); the string carries one or more typed
`Failure` objects rendered to text. Failures are rendered, not raised, so a
soft-mode build can record them and continue.

The architecture is **graph-native**. Both schedules are reduced to a single
`DataflowGraph` (`ScheduleCapture.py:567`) spanning four loop bodies — `ML-1`,
`ML`, `NGL`, `NLL` — whose nodes are real producer/consumer instructions and
whose edges are register or LDS-region dataflow with discrete `edge_kind` tags
(`raw_intrawave`, `lr_to_gr_lds_reuse`, `gr_to_lr_lds_reuse`,
`lds_raw_intrawave`). Validation is two phases:

1. **Cross-graph diff.** `compare_graphs` (`ScheduleCapture.py:3334`) treats
   the default-scheduler graph as canonical and reports any reference edge
   missing from the CMS graph. Each missing edge is classified by
   `diagnose_missing_edge` (`ScheduleCapture.py:3415`) into one of seven
   typed `Failure` subclasses.
2. **Per-edge wait coverage.** `validate_edge_wait_coverage`
   (`ScheduleCapture.py:4364`) walks every edge in the CMS graph and asks
   whether the CMS instruction stream between producer and consumer contains
   an SWaitCnt that drains the producer's counter, an SBarrier where one is
   required, and (for MFMA pipelines) enough quad-cycles for the result to
   become visible.

A new contributor only needs to know two entry points (`isValid`,
`build_dataflow_graph`), seven failure types, and the four-body iteration
order to navigate the rest. The remainder of this doc is the map.

## 2. Goals

The validator exists to catch CMS bugs that would otherwise produce silent
miscompares at runtime. Concretely:

1. **Detect missing dataflow.** Every register or LDS-region edge in the
   default schedule must exist in the CMS schedule, modulo legitimate
   pipelining transformations (cross-subiter ALU producers, cross-body
   handoffs).
2. **Detect missing waits/barriers.** Every dataflow edge must be safe:
   an SWaitCnt drains the right counter to a small-enough value, and (for
   LDS-reuse patterns) an SBarrier follows the wait.
3. **Detect ordering violations.** Same-body producer/consumer order must
   not invert relative to the default schedule.
4. **Detect timing violations** for MFMA → MFMA, CVTPack → MFMA, and
   PackMFMA → CVTPack pairs whose visibility windows are dictated by ISA
   section 7.6 quad-cycle accounting.
5. **Detect SCC clobbers.** A second SCC writer between an SCC producer
   and its consumer is an `OverriddenInputFailure`, not a missing-edge
   miss.
6. **Detect the TF32 middle-pack pair-interleaving invariant.** Each
   `(v_cvt_f32_bf16, v_sub_f32)` pair must be contiguous in the global
   stream order; another `MiddlePack` between them clobbers the temporary
   VGPR.
7. **Surface every defect as a typed Failure** with a stable `format()`
   string so consumers (CI logs, test assertions, future tooling) can
   parse without grepping.

### Out of scope

- **The validator does not generate schedules.** It accepts whatever CMS
  emitted. If the default and CMS captures don't agree on which
  instructions exist (`compare_graphs`'s identity-coverage check at
  `ScheduleCapture.py:3360`), that's a capture-pipeline bug, not a CMS
  defect — the validator raises `CaptureConsistencyError` and gives up.
- **Counter-value verification.** The validator trusts that an `SWaitCnt
  dscnt=N` was emitted with the correct `N`; it does not count outstanding
  loads itself.
- **SBarrier latency.** Treated as instantaneous synchronization.
- **DTL=0 operand rules and WMMA shapes** — RDNA 3.5 (gfx1151) is not
  wired through `_OPERAND_RULES`; see Scope §6 below.
- **LocalSplitU > 1.** Asserted out at `Timeline._populate_instructions`
  (`CMSValidator.py:694`).
- **DirectToLds=0.** Asserted out at the same site
  (`CMSValidator.py:693`). Today every CMS-eligible kernel is DTL=1.
- **Schedule generation, autotuning, kernel selection.** All upstream of
  the validator.

## 3. High-level approach

Two passes run back-to-back in `isValid` once both default and CMS captures
are present (`CMSValidator.py:1089`):

```
KernelWriter -> customMainLoopSchedule -> ScheduleInfo + idMap
       |                                           |
       v                                           v
 default-side capture                         cms-side capture
 (SIA3 replay)                                (CMS emission)
       \                                       /
        \                                     /
         v                                   v
        FourPartCapture(default)     FourPartCapture(cms)
                |                            |
                v                            v
        build_dataflow_graph         build_dataflow_graph
                |                            |
                v                            v
            ref_graph                    subj_graph
                  \         |         /
                   \        |        /
                    v       v       v
              compare_graphs(ref, subj)        validate_edge_wait_coverage(subj)
                    |                                    |
                    v                                    v
              List[Failure]                       List[Failure]
                       \                          /
                        \                        /
                         v                      v
                  isValid(...) -> (False, "<rendered failures>")
```

Each `FourPartCapture` (`ScheduleCapture.py:299`) is a symmetric per-codepath
container holding the four scheduled bodies — `main_loop_prev`, `main_loop`,
`n_gl`, `n_ll` — plus `num_mfma`, `num_codepaths`, the per-arch
`ArchProfile`, and the inner-unroll subiteration count. The graph builder
walks the bodies in execution order
(`_BODY_BUILD_ORDER = (ML-1, ML, NGL, NLL)` at
`ScheduleCapture.py:2961`) and produces one node per real instruction and one
node per scheduler control op (SWait, SBarrier, SNop). Only the real
instructions enter `nodes_by_identity`; control ops live in the per-body
sidecar (`body._graph_nodes`) so the wait/barrier walkers can find them in
stream order.

The graph is **unified** — one graph spans all four bodies. Cross-body edges
(DTL+LdsBuf: an LR0 in ML-1 feeding a GR in ML) are first-class and carry
producer / consumer `body_label`s that differ. The cross-body case is not
special-cased in the comparison or coverage passes; it just shows up as a
larger position delta.

## 4. Invariants leveraged

These are the assumptions baked into the architecture. Each is enforced
somewhere in code; if it were violated the listed consequence would follow.

### 4.1 The default schedule is the canonical reference

The default scheduler (SIA3) is treated as authoritatively correct.
`compare_graphs` reports edges *missing from the CMS graph that are present
in the default graph*; `OrderInvertedFailure`
(`ScheduleCapture.py:731`) only fires when the default emitted producer
before consumer and CMS emitted them in the opposite relative order.

- Enforced at: `compare_graphs` (`ScheduleCapture.py:3399`) builds
  `missing_keys = ref_keys - subj_keys`.
- Consequence if violated: a CMS reorder that's actually a bug becomes
  invisible when the default also reorders, OR a legitimate CMS reorder is
  flagged when the default's resolver emitted an artifactual edge. In
  practice the resolver's per-byte latest-writer pass
  (`ScheduleCapture.py:3121`) makes both sides emit deterministic edges,
  so this is stable.

### 4.2 Bodies execute back-to-back; the MFMA pipeline state carries through

`cumulative_issue_cycles` (`ScheduleCapture.py:3963`) walks instructions
across body boundaries in `_BODY_BUILD_ORDER` order. The simulator state
(`current_issue`, `mfma_free_at`, `last_mfma_class`, `last_mfma_issue`)
**persists across body boundaries** because the hardware doesn't reset at
source-code body edges.

- Enforced at: `cumulative_issue_cycles` lines 4060–4144.
- Consequence if violated: cross-body MFMA → MFMA gap calculations would
  be off; PackMFMA → CVTPack timing failures could miss real defects or
  fire on safe schedules. A previous `body_delta * 1000` placeholder
  always returned ok=True; the simulator banishes it (note in the
  `_quad_cycle_gap_ok` docstring at `ScheduleCapture.py:4166`).

### 4.3 Every captured non-control instruction has a `tagged_inst`

`GraphNode` (`ScheduleCapture.py:517`) carries `tagged_inst:
TaggedInstruction` as a back-reference to the captured stream entry. The
formatter helpers (`cms_node_label` at `ScheduleCapture.py:934`,
`_node_position` at `ScheduleCapture.py:646`) and the
per-category-stream `[N]` index lookup all read through `tagged_inst`.

- Enforced at: `_make_node` (`ScheduleCapture.py:2938`) requires the
  tagged instruction; `LoopBodyCaptureBuilder.finalize`
  (`ScheduleCapture.py:1075`) raises `CaptureWiringError` if any
  TaggedInstruction has `inst=None`.
- Consequence if violated: the formatter falls through to a bare-category
  primary (`PackA0` instead of `PackA0[3]`), losing the `[N]` disambiguation.
  Tests in `test_failure_formatters.py` would notice immediately.

### 4.4 LCC instructions are graph nodes; they cost real cycles

LCC (loop-counter-control) emits `SSubU32 + SCmpEQI32` per body, costing
2 quad-cycles per body. They are first-class graph nodes — their
per-instruction issue cycles contribute to `cumulative_issue_cycles` walks.

- Enforced at: `build_dataflow_graph` constructs LCC nodes alongside
  every other instruction (`ScheduleCapture.py:3055`); the per-body
  `\useLoop` macro guard suppresses LCC in NGL/NLL (see the `LCC_AUDIT.md`
  line-110 conclusion). The exclusion `node.identity[0] != "LCC"` cited
  in `LCC_AUDIT.md` Step 5 has since been dropped — LCC participates in
  the cross-graph identity set.
- Consequence if violated: cross-body cycle counts would underestimate by
  ~2 quad-cycles per body, weakening the timing checks.

### 4.5 The dscnt FIFO model excludes SMEM, flat, and vector stores

The per-counter wait model used by `validate_edge_wait_coverage` assumes
each LR/LW/GR enqueues exactly one entry on a single counter. SMEM ops
also decrement dscnt; flat ops decrement two counters; vector stores use
vscnt (which is not modeled).

- Enforced at: `LoopBodyCaptureBuilder.finalize`
  (`ScheduleCapture.py:1082, 1089, 1096`) raises
  `CaptureSMEMError` / `CaptureFlatError` / `CaptureStoreError` on
  capture if any forbidden class appears.
- Consequence if violated: the wait-coverage classifier would
  mis-attribute counter drains and emit either false-positive or
  false-negative `MissingWaitFailure` / `WaitInsufficientFailure`.

### 4.6 SchedulePosition orders unambiguously across bodies

`SchedulePosition` (`ScheduleCapture.py:480`) is `(loop_index,
vmfma_index, sub_index)` with `loop_index` derived from `body_label`
through `BODY_LABEL_TO_LOOP_INDEX` (`ScheduleCapture.py:470`). The
`@functools.total_ordering` + `__lt__` gives a total order over every
node in the graph.

- Enforced at: tuple-comparison in `__lt__`
  (`ScheduleCapture.py:493`); used by `cumulative_issue_cycles`
  (`ScheduleCapture.py:4021`) to gate "producer before consumer".
- Consequence if violated: cross-body edges would have undefined order;
  the FIFO simulator would fall back to the defensive 0-cycle return.

### 4.7 The capture-pipeline is consistent with `kernel_emits_n_gl` /
`kernel_emits_n_ll`

A `FourPartCapture` may legitimately have an absent body (PGR /
SuppressNoLoadLoop combinations skip NGL or NLL). `assert_capture_body_consistency`
(`ScheduleCapture.py:4901`) raises `CaptureConsistencyError` if the
captured-body presence doesn't match the predicates derived from kernel
config — that signals either a production-gate change in
`KernelWriter.kernelBody` not mirrored in the predicates, or a
capture-pipeline bug.

- Consequence if violated: `build_dataflow_graph` would silently skip a
  body the validator should have walked, masking real defects.

## 5. Assumptions made

These are inputs the validator requires but does not enforce. If you
violate them the output is undefined.

- **`idMap` is complete.** `assert_idmap_completeness`
  (`ScheduleCapture.py:1353`) is the explicit check, but it only fires
  if `build_idmap` was called via the canonical wiring. A test fixture
  that constructs a partial idMap and skips the assertion will get
  silently truncated graphs.
- **Default-side and CMS-side captures share the same
  `num_mfma_per_subiter`.** The graph derives subiter membership via
  `vmfma_index // num_mfma_per_subiter`
  (`ScheduleCapture.py:1990`). If one side passes 0 and the other
  passes the real value, cross-subiter MFMA edges will compare unequal
  when they're really the same.
- **Captures are bodies of the SAME kernel.** No invariant guards against
  passing two unrelated FourPartCaptures into `compare_graphs`.
- **Every `rocisa_inst` has a recognized class.** `_class_tag`
  (`ScheduleCapture.py:1844`) and the operand-rule registry
  (`_DSLoadRule` ... `_GenericALURule` at
  `ScheduleCapture.py:2168–2749`) cover every class production currently
  emits. A new instruction class added without a registry entry raises
  `CaptureUnknownInstructionError` at `build_dataflow_graph`
  (`ScheduleCapture.py:3060`).
- **Bodies are non-empty.** A present-but-empty body raises
  `CaptureEmptyBodyError` (`ScheduleCapture.py:3032`); an absent body is
  treated as "this body was not emitted." See the comment at
  `ScheduleCapture.py:3009`.
- **`ArchProfile` lookup falls back to CDNA 4.** Unknown ISAs get the
  default profile silently (`ScheduleCapture.py:431`). New archs need a
  profile registered in `_ARCH_PROFILES_BY_ISA`
  (`ScheduleCapture.py:419`); without one the timing constants are wrong
  and the validator silently passes schedules that violate the new
  arch's visibility windows.
- **The captured stream is in execution order.** No invariant
  re-orders; the graph builder reads `body.instructions` linearly
  (`ScheduleCapture.py:3055`).

## 6. Scope

| Covered | Not covered |
|---|---|
| CDNA 4 (gfx950), DTL=1, LocalSplitU=1 | CDNA 4 with DTL=0 (asserted out at `CMSValidator.py:693`) |
| Single-codepath and 2-codepath CMS schedules | LocalSplitU > 1 (asserted out at `CMSValidator.py:694`) |
| BF16, F8, F32, dot2, sparse, conversion variants | RDNA 3.5 (gfx1151): `Timeline` bypassed; graph runs but `_OPERAND_RULES` not wired for DTL=0/WMMA |
| TF32 emulation (24-pack and 4x4 PackMFMA chains) | gfx940/941/942/90a (no CMS schedules registered) |
| LR/LW/GR/MFMA register dataflow | Vector stores (no current CMS body emits them) |
| LDS-region dataflow (`lds_raw_intrawave`) | SMEM ops (capture rejects them — see invariant 4.5) |
| LR → SBarrier → GR LDS-reuse patterns | Flat ops (capture rejects) |
| GR → SBarrier → LR LDS-reuse patterns | SBarrier latency (treated as instant) |
| SCC dataflow + SCC clobber detection | SWaitCnt counter-value verification (trusted, not recomputed) |
| Quad-cycle timing for MFMA→MFMA, CVTPack→MFMA, PackMFMA→CVTPack | SCC chaining wait-states (LCC instructions default to 1 quad-cycle each per `LCC_AUDIT.md` §6.1) |
| MFMA type-switch +1 stall | `ForceUnrollSubIter` register reuse (partial — see TODO at `CMSValidator.py:748`) |
| TF32 middle-pack pair-interleaving | Schedule generation / autotuning |
| Cross-body cycle accounting in `_BODY_BUILD_ORDER` | |

## 7. Key technical details

The non-obvious mechanics. Each is something a reader hitting the live code
will want explained before they can navigate.

### 7.1 The two entry points in `isValid`

`isValid` (`CMSValidator.py:1058`) does very little structurally. It pulls
the default and CMS captures off `ValidationContext`, attaches the per-arch
`ArchProfile` (`CMSValidator.py:1098`), builds two graphs, and runs
`compare_graphs` then `validate_edge_wait_coverage`. The historical
`TIMELINE_RULES` and `STRUCTURAL_RULES` lists have been emptied; the only
remaining structural-side checks are the per-instruction `validate()`
overrides on `GlobalRead.validate` (`CMSValidator.py:308`) and
`SWait.validate` (`CMSValidator.py:406`), each of which emits typed
Failures. `findValidPositions` (`CMSValidator.py:1135`) is a diagnostic
helper that brute-forces every legal vmfma slot for an instruction.

### 7.2 `cumulative_issue_cycles` is the one-and-only cycle simulator

Every quad-cycle gap check (`_quad_cycle_gap_ok`,
`_cvt_to_mfma_gap_ok`, `_mfma_pack_to_cvt_gap_ok` at
`ScheduleCapture.py:4151, 4192, 4226`) delegates to
`cumulative_issue_cycles` (`ScheduleCapture.py:3963`). The function walks
the unified instruction stream from producer to consumer across body
boundaries, accumulating per-instruction issue costs, MFMA finish-time
stalls, and the type-switch +1 penalty. A previous structural-side mirror
(`precompute_issue_times`) was removed; this is now the canonical source.
Same-body and cross-body share one code path (`ScheduleCapture.py:4060`):
the cross-iteration distinction is a red herring because the graph lays
out instructions in execution order regardless of body membership.

### 7.3 The seven Failure subclasses

All in `ScheduleCapture.py:684` (`Failure` base) plus subclasses:

- **`OrderInvertedFailure`** (`:731`) — same-body producer issued after
  consumer in CMS but before in default. Suppressed for cross-subiter ALU
  producers (legitimate pipelining; gate at `:3475`).
- **`MissingWaitFailure`** (`:755`) — no SWaitCnt drains the producer's
  counter in the producer→consumer window. Carries
  `nearby_wait_indices` for SWaits on other counters.
- **`WaitInsufficientFailure`** (`:784`) — SWaitCnt exists but its
  counter value doesn't drain the producer's queue position.
- **`MissingBarrierFailure`** (`:817`) — wait covers but no SBarrier
  follows for an LDS-reuse edge.
- **`TimingTooCloseFailure`** (`:845`) — not enough quad-cycles between
  MFMA / CVTPack / PackMFMA producer and consumer.
- **`InvalidCounterValueFailure`** (`:866`) — an SWaitCnt's `dscnt` /
  `vlcnt` / `vscnt` is out of range. Emitted by `SWait.validate`
  (`CMSValidator.py:406`).
- **`OverriddenInputFailure`** (`:902`) — a second writer between
  producer and consumer clobbered the input register (SCC clobber, or
  TF32 middle-pack pair-interleave violation).

Each Failure carries pre-rendered `FailureNodeLabel`s
(`ScheduleCapture.py:654`), an `iter_delta` for cross-iteration suffix
rendering, and zero rocisa back-references — the Failure is a
data-only artifact safe to log, pickle, or assert against. Wording lives
in each subclass's `_format_canonical()`.

### 7.4 The `_node_label` / `cms_node_label` contract

Every formatter call goes through `cms_node_label`
(`ScheduleCapture.py:934`). It accepts both `GraphNode` (graph-side) and
`ValidatorInstruction` (structural-side, e.g. inside `GlobalRead._validate_needed_by`
at `CMSValidator.py:318`) via the `NodeLike` type alias
(`ScheduleCapture.py:50`). The discriminator is a getattr probe — both
shapes carry `category` (one as a property, one as a field) and the helper
walks the body's TaggedInstructions to compute the per-category `[N]`
index. If `tagged_inst` is missing, the label falls back to bare-category
primary; that's why invariant 4.3 matters.

### 7.5 The four pair-specific quad-cycle helpers

The CDNA 4 ISA section 7.6 visibility windows live as four constants on
`ArchProfile` (`ScheduleCapture.py:353`):

- `standard_mfma_finish_cycles = 3` — full-tile MFMA finish.
- `mfma_4x4_finish_cycles = 1` — 4x4 PackMFMA finish.
- `cvt_before_mfma_quad_cycles = 2` — CVTPack→MFMA settle.
- `mfma_4x4_before_cvt_quad_cycles = 5` — PackMFMA→CVT1 settle.

Three pair-specific helpers consult them: `_quad_cycle_gap_ok`
(`:4151`) for MFMA→consumer, `_cvt_to_mfma_gap_ok` (`:4192`) for
CVTPack→MFMA, and `_mfma_pack_to_cvt_gap_ok` (`:4226`) for
PackMFMA→CVTPack. The dispatch order in `_classify_edge_coverage`
(`:4438`, `:4458`, `:4483`) is load-bearing: PackMFMA producers must
hit `_mfma_pack_to_cvt_gap_ok` before the generic `_is_mfma_producer`
branch, and CVTPack producers must hit `_cvt_to_mfma_gap_ok` before the
ALU-immediate exemption (`_is_alu_producer` at `:4274`) absorbs them.
Each carve-out is documented in-line because the dispatch interactions
are the failure mode every refactor of this code keeps reintroducing.

### 7.6 The per-byte latest-writer resolver

`build_dataflow_graph`'s phase 2 (`ScheduleCapture.py:3098–3153`) walks
the unified data-flow nodes in stream order and maintains a
`latest_writer` map from `byte_key` to `(writer_node, write_resource)`.
Reads emit one edge per distinct prior writer; writes overwrite the map.
A new write to a byte key OVERWRITES the previous writer — exactly
"latest writer", and the fix for the phantom-edge bug from scratch-VGPR
reuse. **No subiter scoping**: a vgpr is one physical register. If a
kernel mis-pipelines a prefetch (PackA1 writes v133 before PackA0's
subiter-0 consumer reads it), the resolver faithfully reports PackA1 as
the producer — the same garbage value the GPU will read. Adding subiter
scoping would HIDE such bugs to make diagnostics look cleaner; the
in-line comment at `:3115–3120` makes the trade-off explicit.

### 7.7 SBarrier-pattern collector for LDS reuse

`_collect_barrier_edges` (`ScheduleCapture.py:3188`) and `_collect_pattern`
(`ScheduleCapture.py:3230`) sweep the unified node stream once per
direction:

- `lr_to_gr_lds_reuse`: LR0/LR1 → SWaitCnt(dscnt) → SBarrier → GR.
- `gr_to_lr_lds_reuse`: GR → SWaitCnt(vlcnt) → SBarrier → LR1/LR3.

Both demand strict ordering. The state machine restarts on a new
producer of the same kind. Cross-body patterns form naturally because
the sweep is over `nodes_per_body` concatenated in `_BODY_BUILD_ORDER`;
the resulting `DataflowEdge`'s `producer.body_label` and
`consumer.body_label` may differ.

### 7.8 The `raise_on_unexplained` knob

Both `compare_graphs` and `validate_edge_wait_coverage` accept a
`raise_on_unexplained` parameter. Tests pass `True` so any classifier
fall-through raises `UnexplainedMissingEdgeError`
(`ScheduleCapture.py:104`). Production calls from `isValid` pass
`False` (`CMSValidator.py:1108, 1120`) so an unclassified edge becomes a
soft Failure instead of crashing the build.

### 7.9 SCC clobber path

`diagnose_missing_edge`'s SCC handling
(`ScheduleCapture.py:3510–3539`) is structurally distinct from
register-dataflow handling. SCC is single-bit and not preserved across
loop iterations by any compiler convention, so cross-body SCC edges in
the default graph are aliasing artifacts of the per-byte resolver
running over the SCC sentinel — the path explicitly suppresses them
(`:3517`). Same-body SCC misses look for an intervening SCC writer in
the subject graph and emit `OverriddenInputFailure` carrying the
producer/consumer/clobber triple.

### 7.10 The middle-pack pair-interleaving check is stream-shape, not
edge-shape

`validate_middle_pack_pair_interleaving`
(`ScheduleCapture.py:4600`) does not run per-edge — it walks every
MiddlePack node in the unified stream, buckets by category, pairs them
by adjacency within their category-local list, and confirms that no
other MiddlePack sits between a leader and its pair-consumer in the
GLOBAL stream order. A violation emits `OverriddenInputFailure` with
`resource="vgpr"` and the intervening node as the clobber.

## 8. Future work

The `br4` consolidation epic (`Tensile/Components/CMS_VALIDATOR_CONSOLIDATION_AUDIT.md`)
is moving the validate-time bulk of `ScheduleCapture.py` (~2200 lines:
`Failure` classes, `ArchProfile`, `GraphNode` / `DataflowEdge` /
`DataflowGraph`, `build_dataflow_graph`, `compare_graphs`,
`diagnose_missing_edge`, the FIFO simulator, `cumulative_issue_cycles`,
the four pair-specific helpers, `validate_edge_wait_coverage`,
`validate_middle_pack_pair_interleaving`) into `CMSValidator.py`,
leaving the capture file scoped to capture machinery only. Sub-beads
br4.3..br4.10 (per-stage moves and a final hygiene pass) remain open.
Adjacent work: `5gd` (per-byte latest-writer regressions), `nn0`
(methods-on-classes), `c70` (Register abstraction), `wx9.3` (gfx1151
operand-rule wiring for DTL=0). The `09y` naming pass will rename both
files post-consolidation. Bead `2yg` is repurposed for the final design-doc
pass once all of the above land.
