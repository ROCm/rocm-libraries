# `Timeline` / `TimelineEvent` interface spike (rocm-libraries-cil)

Design spike for the input shape of the generalized validator (parent bead
`rocm-libraries-5gd` part A; sibling refactor bead `rocm-libraries-3g4`).

This document is doc-only. No production Python is written, no test fixtures
are converted in code, no parallel APIs are introduced. The conversion shown
in §4 is illustrative — the actual implementation belongs to bead
`rocm-libraries-xe5` (`cms_capture_to_timeline`) and bead
`rocm-libraries-x6s` (`assembly_to_timeline`).

The design walks bottom-up from the consumers' actual needs, not top-down
from a hypothetical asm-flavored API. Anything invented "for later" is out of
scope per the parent bead's "no fallback paths, no parallel APIs" directive.

---

## 1. Consumer audit — what the validator actually reads today

All callsites below are in
`projects/hipblaslt/tensilelite/Tensile/Components/CMSValidator.py` unless
noted. `ScheduleCapture.py` is the producer side; references there are
included where they describe the consumed shape rather than its production.

### 1.1 `FourPartCapture` field accesses

| File:Line | Field | Use |
|---|---|---|
| CMSValidator.py:935 | `four_part_capture.main_loop_prev` | seed `captures[BODY_LABEL_ML_PREV]` in `build_dataflow_graph` |
| CMSValidator.py:936 | `four_part_capture.main_loop` | seed `captures[BODY_LABEL_ML]` |
| CMSValidator.py:937 | `four_part_capture.n_gl` | seed `captures[BODY_LABEL_NGL]` |
| CMSValidator.py:938 | `four_part_capture.n_ll` | seed `captures[BODY_LABEL_NLL]` |
| CMSValidator.py:951 | `four_part_capture.num_mfma_per_subiter` (via `getattr`) | propagated to `DataflowGraph.num_mfma_per_subiter`; used by `_node_subiter` and the MFMA-pair classifiers |
| CMSValidator.py:956 | `four_part_capture.arch_profile` (via `getattr`) | propagated to `DataflowGraph.arch_profile`; used by all timing helpers and `_make_node`'s `issue_cycles` populator |
| CMSValidator.py:4041 | `context.default_capture.arch_profile` | branch in `kernelBody`'s wrapper to populate the profile if absent (a writer-side concern, not a consumer of the body shape itself) |

`num_mfma`, `num_codepaths`, and `source` on `FourPartCapture` are NOT read
anywhere in `CMSValidator.py`. Confirmed by a grep over the Components
directory; they are write-only metadata as of today's tip
(`bb96c76a19`). They are kept on the producer side for capture-pipeline
guards (`expand_cms_macro` consistency checks, `CmsCaptureInputs` plumbing)
but never feed the validator.

The four body slots are read with their indexed shape `{0: body}` because the
graph builder has the per-codepath dict baked in. Only `body = by_cp[0]` is
ever consumed (CMSValidator.py:943); the multi-codepath dict shape is purely
producer-side bookkeeping.

### 1.2 `LoopBodyCapture` field accesses

| File:Line | Field | Use |
|---|---|---|
| CMSValidator.py:944 | `body.instructions` (truthiness check) | non-empty-body precondition (`CaptureEmptyBodyError`) |
| CMSValidator.py:969 | `body.instructions` | iterate `tagged_inst` to build per-body `_graph_nodes` |
| CMSValidator.py:1003 | `body._graph_nodes = ...` | per-body sidecar attached **by** `build_dataflow_graph` so cross-body walkers (`_all_nodes_in_order`, `waits_in_window`, `barriers_in_window`) can find sync ops in stream order. This is a write the graph-builder PERFORMS on the body, not a field the body provides; the Timeline equivalent will need a sidecar slot or a separate `Timeline.events_by_body` accessor. |
| CMSValidator.py:1054 | `body_capture.name_to_idx` (via `getattr`) | per-body symbolic-name → numeric-base map; forwarded to `_byte_keys_for_resource` and `_resolve_producers` so symbolic vgpr operands collapse to the same byte-key as numeric writes of the same physical register (rocm-libraries-bb34) |
| CMSValidator.py:545 | `cap._graph_nodes` (via `hasattr` + iteration) | `_all_nodes_in_order` enumerates per-body nodes in stream order |
| CMSValidator.py:2310 | `body_capture.instructions` | `cms_node_label` builds a per-category-stream 0-based index by indexing `[t for t in body_capture.instructions if t.category == cat]`. **CMS-specific**: see §5. |

### 1.3 `TaggedInstruction` field accesses

| File:Line | Field | Use |
|---|---|---|
| CMSValidator.py:835 | `tagged_inst.wrapped.rocisa_inst` | underlying rocisa instance for `_identity_for` / `_canonical_render` / class-tag dispatch |
| CMSValidator.py:836 | `tagged_inst.category` | `_identity_for` discriminator (e.g. PackMFMA vs main-loop MFMA), threaded into `GraphNode.category`, the cross-graph identity tuple, and downstream `_is_*_producer` predicates |
| CMSValidator.py:837 | `tagged_inst.slot` | `make_position(body_label, tagged_inst.slot)` → `SchedulePosition(loop_index, vmfma_index=slot.mfma_index, sub_index=slot.sequence)` |
| CMSValidator.py:838 | `tagged_inst.category` | `name = f"{category}@{vmfma_index}.{sub_index}"` — human-facing label (`GraphNode.name`) |
| CMSValidator.py:848 | `tagged_inst.category` | stored directly on `GraphNode.category` |
| CMSValidator.py:850 | `tagged_inst` | back-reference stashed on `GraphNode.tagged_inst` for downstream consumers |
| CMSValidator.py:976 | `tagged_inst.category` | error-message context in `CaptureUnknownInstructionError` |
| CMSValidator.py:1043 | `node.tagged_inst.wrapped` | reaches `wrapped.reads`, `wrapped.writes`, `wrapped.read_slots`, `wrapped.write_slots` for edge formation |
| CMSValidator.py:2306 | `node.tagged_inst` (via `getattr`) | `cms_node_label` looks it up to find the per-category-stream index |
| CMSValidator.py:2311 | `t.category` (via `getattr`) on `body_capture.instructions` element | filter for the `same_cat` group |
| CMSValidator.py:2314 | `same_cat.index(tagged)` | the per-category-stream `[N]` index for the `LRA0[3]` rendering |

### 1.4 `WrappedInstruction` field accesses (via `tagged_inst.wrapped`)

| File:Line | Field | Use |
|---|---|---|
| CMSValidator.py:1043, 1060–1063 | `wrapped.reads`, `wrapped.read_slots` | per-operand read resources + their positional operand-slot indices for Phase-2 read edges |
| CMSValidator.py:1087–1090 | `wrapped.writes`, `wrapped.write_slots` | per-operand write resources + slots for Phase-2 latest-writer updates |
| CMSValidator.py:1043 | `wrapped` itself | passed by reference into the resolver loop |

`reads`/`writes` are tuples of `RegisterContainer | MemoryRegion | None`;
`read_slots`/`write_slots` are tuples of small integers parallel to them.
Population is producer-side via `_populate_wrapper`
(ScheduleCapture.py:1632) which dispatches an operand-rule registry
keyed on the rocisa class. From the consumer's point of view these are
opaque pre-populated tuples.

### 1.5 `SchedulePosition` field accesses (already present on `GraphNode.position`)

| File:Line | Field | Use |
|---|---|---|
| CMSValidator.py:752 | `node.position.vmfma_index // num_mfma_per_subiter` | MFMA subiter derivation |
| CMSValidator.py:1040 | `n.position` (sort key) | stream-order traversal of all data-flow nodes |
| CMSValidator.py:601 | `start < node.position < end` | window membership for `barriers_in_window` |
| CMSValidator.py:572–584 | similar window comparisons | `waits_in_window` window membership + drain check |
| CMSValidator.py:2327 | `pos.vmfma_index` | `cms_node_label` position string `@ idx={vmfma_index}` |
| CMSValidator.py:2342 | `c_pos.loop_index - p_pos.loop_index` | `_cms_iter_delta` for `Failure._iter_suffix` |
| CMSValidator.py:2530–2532 | `ref_p.position < ref_c.position` | order-inversion check in `diagnose_missing_edge` |
| CMSValidator.py:2992 | `_node_position(insufficient).vmfma_index` | reported `wait_idx` on `WaitInsufficientFailure` |

The `<` operator on `SchedulePosition` uses tuple-style lex sort
`(loop_index, vmfma_index, sub_index)`, matching the global stream order
across bodies.

### 1.6 `SlotKey` field accesses

| File:Line | Field | Use |
|---|---|---|
| ScheduleCapture.py:426–427 (consumer shape via `make_position`) | `slot.mfma_index`, `slot.sequence` | mapped into `SchedulePosition.vmfma_index` / `.sub_index` |

`SlotKey.subiter` and `SlotKey.slot_kind` are NOT read by `make_position` and
are NOT consumed downstream by `CMSValidator.py`. They are producer-side
(used by `LoopBodyCaptureBuilder` to discriminate pre-loop / mfma-slot /
post-loop buckets at finalize time and for `expand_cms_macro` consistency
checks).

### 1.7 Summary — distinct fields the validator actually consumes

Per-event:

  * `wrapped.rocisa_inst` (the raw instruction; opaque from the validator's
    POV except for `_class_tag`, `_canonical_render`, and the rocisa
    `getattr` probes inside `_swait_drains`/`_collect_pattern`)
  * `wrapped.reads`, `wrapped.writes` — tuples of `RegisterContainer |
    MemoryRegion | None`
  * `wrapped.read_slots`, `wrapped.write_slots` — tuples of small int
  * `category` — string scheduler-role tag
  * `slot.mfma_index`, `slot.sequence` — coordinates that fold into
    `SchedulePosition`
  * `body_label` — string in `{ML-1, ML, NGL, NLL}`, supplied by the body
    the event lives on (not by the event itself in today's shape)

Per-body:

  * the ordered sequence of events
  * `name_to_idx` (per-body symbolic-name → numeric-base map; CMS-specific
    today, see §5)
  * a sidecar slot the graph builder writes (`_graph_nodes`) for
    cross-body stream walks

Per-Timeline (kernel-wide):

  * `arch_profile` — `Optional[ArchProfile]` for timing helpers
  * `num_mfma_per_subiter` — `int` for MFMA subiter derivation

That is the complete consumed surface as of `bb96c76a19`. Everything else
on `FourPartCapture`/`LoopBodyCapture`/`TaggedInstruction`/`SlotKey` is
producer-side bookkeeping or write-only metadata.

---

## 2. `TimelineEvent` — the minimal source-agnostic shape

Below is the proposed dataclass shape, in pseudo-Python (NOT to be
implemented yet). Each field maps directly to a §1 row.

```python
@dataclass
class TimelineEvent:
    """One scheduled instruction in source-agnostic form.

    Carries exactly what the validator's graph builder + edge-coverage logic
    consumes today. No fields beyond the §1.7 catalogue.
    """
    # Underlying rocisa Instruction (opaque to the validator except for the
    # _class_tag / _canonical_render / rocisa-attr probes already in place).
    # For an asm-source Timeline this is still a rocisa Instruction —
    # `assembly_to_timeline` (rocm-libraries-x6s / rocm-libraries-7qm) is
    # responsible for materializing one. Keeping the field's type identical
    # across sources avoids a discriminator branch in `_class_tag`.
    rocisa_inst: object

    # Pre-populated reads / writes + parallel positional operand-slot indices.
    # Same shape and semantics as `WrappedInstruction.{reads, writes,
    # read_slots, write_slots}` today. The asm bridge populates these from
    # the rocisa instance via the existing _populate_wrapper rule registry
    # (rocm-libraries-7qm).
    reads: tuple             # tuple[RegisterContainer | MemoryRegion | None, ...]
    writes: tuple            # tuple[RegisterContainer | MemoryRegion | None, ...]
    read_slots: tuple        # tuple[int, ...] parallel to reads
    write_slots: tuple       # tuple[int, ...] parallel to writes

    # Scheduler-role tag. CMS supplies categories like "LRA0", "PackB1",
    # "MFMA", "SYNC". An asm-source Timeline supplies tags from a
    # source-aware classifier (rocm-libraries-3dy / 5gd.B.1). The string
    # itself remains the discriminator for `_class_tag_from_category`,
    # `_is_alu_producer`, etc.; the asm tag schema must therefore preserve
    # the prefixes those predicates pattern-match on (LR, LW, GR, MFMA,
    # Pack, SYNC, BARRIER, SNOP, SSETPRIO, LCC).
    category: str

    # Stream-position coordinates within the body. See §3 for the position
    # discussion and the open question on whether to keep SchedulePosition
    # as-is or generalize to a flat (body_label, stream_index) shape.
    position: SchedulePosition

    # Body the event lives on. Today this is implicit (bodies are dict
    # values; `body_label` is set on `GraphNode` by `_make_node` from the
    # for-loop iterator over `_BODY_BUILD_ORDER`). Lifting it onto the
    # event makes the per-event source-agnostic shape closed: the consumer
    # never has to ask "which container did I come from?".
    body_label: str
```

### Notes on what is intentionally NOT on `TimelineEvent`

  * No `wrapped` indirection. Today `TaggedInstruction.wrapped` is a
    `WrappedInstruction` proxy because rocisa-bound classes refuse
    `setattr` and need an external place to hold `reads`/`writes`. The
    `TimelineEvent` dataclass owns those fields directly, so the proxy
    layer disappears at the consumer boundary. (The producer-side
    `WrappedInstruction` may still exist as the in-memory carrier during
    capture; the bridge functions copy out the four tuples and discard
    the wrapper.)
  * No `slot: SlotKey`. `SlotKey.subiter` and `SlotKey.slot_kind` are
    producer-side and never reach the validator. `mfma_index` and
    `sequence` survive only as `SchedulePosition.vmfma_index` and
    `.sub_index` — already on `position`.
  * No back-reference to a "body" object. The graph builder uses
    `body_label` to look the body up in a dict; flattening that lookup
    onto the event removes the indirection.

---

## 3. `Timeline` — positionable sequence of events

### 3.1 Recommendation — per-body dict, NOT flat list

```python
@dataclass
class Timeline:
    """Source-agnostic input to the validator.

    Holds the per-body event streams plus the kernel-wide knobs the
    validator needs.
    """
    # Per-body event streams, keyed by body_label. The four CMS body labels
    # ("ML-1", "ML", "NGL", "NLL") are the production set today; an
    # asm-source Timeline emits the same label set even when its source
    # didn't structurally distinguish them — see §3.3.
    events_by_body: Dict[str, List[TimelineEvent]]

    # Kernel-wide knobs.
    arch_profile: Optional[ArchProfile] = None
    num_mfma_per_subiter: int = 0
```

**Rationale for per-body over flat:**

  * `build_dataflow_graph` already iterates per body (`for label in
    _BODY_BUILD_ORDER: body = captures[label]`). A flat
    `List[TimelineEvent]` would force the graph builder to either (a)
    re-bucket by `body_label` at entry — adding O(N) work and a temporary
    dict for nothing, or (b) thread per-body state through a single
    streaming pass, which is a much bigger refactor than this spike's
    parent (`rocm-libraries-3g4`) wants to take on.
  * `_all_nodes_in_order` (CMSValidator.py:534) already uses per-body
    iteration with the `_BODY_BUILD_ORDER` constant. Per-body input lines
    up with how the consumer structures its global stream walks.
  * The per-body sidecar slot (`_graph_nodes`) is attached by the graph
    builder to the body, not to individual events
    (CMSValidator.py:1003). A flat input would force a parallel
    `Dict[body_label, List[GraphNode]]` to live somewhere. Per-body input
    keeps the sidecar attachment shape intact.
  * The body dict can grow `name_to_idx` as a sibling map without
    overloading the event shape — see §3.4.
  * Migration churn: `compare_graphs`, `_body_for_node`,
    `validate_edge_wait_coverage`, and `cms_node_label` all index into
    `graph.captures[body_label]` today. A per-body Timeline preserves
    that access pattern; a flat Timeline forces all four to grow a
    re-bucketing helper.

The cost is a tiny amount of access-pattern noise (`timeline.events_by_body[label]`
vs `timeline.events`) and one extra `if label in timeline.events_by_body` check
per body. That is dramatically less churn than the flat shape would impose.

### 3.2 Position abstraction — keep `SchedulePosition` as-is

`SchedulePosition(loop_index, vmfma_index, sub_index)` is consumed in three
ways today:

  1. As a sort key for stream-order traversal (CMSValidator.py:1040). The
     tuple-style lex order encodes "global stream position across all four
     bodies".
  2. As a comparable-window endpoint (`waits_in_window`,
     `barriers_in_window`, `start < pos < end`).
  3. For diagnostic rendering: `vmfma_index` becomes `@ idx=N` in
     `cms_node_label`, and `loop_index` deltas drive `_cms_iter_delta` /
     `Failure._iter_suffix`.

For an asm-source Timeline, the question is whether `(loop_index,
vmfma_index, sub_index)` survives.

  * `loop_index` is the body's encoding (`BODY_LABEL_TO_LOOP_INDEX`). An
    asm Timeline that only has one body (e.g. a flat asm dump with no
    loop body discrimination) can use `loop_index = 0` for everything —
    or, if the asm-side Timeline distinguishes ML-1/ML/NGL/NLL by
    pre-loop / main / post-loop / cleanup heuristics, it can use the
    same `BODY_LABEL_TO_LOOP_INDEX` map. Either way the field carries
    real ordering information.
  * `vmfma_index` — for an asm Timeline this can be the global stream
    index (asm dump line number divided by some normalization). For
    diagnostic rendering `@ idx=42` reads sensibly as "stream slot 42".
    The bridge bead (rocm-libraries-x6s) is responsible for picking the
    derivation; from the consumer's perspective it just needs a
    stream-monotonic int.
  * `sub_index` — tie-breaker among events sharing a `(loop_index,
    vmfma_index)`. Asm Timeline can keep this as the per-stream-slot
    sequence number, or set it identically to `vmfma_index` and use
    `sub_index = 0` always (if the asm bridge gives every event a
    distinct `vmfma_index`).

Recommendation: **keep `SchedulePosition` as-is**. The shape is generic
enough (a 3-tuple of ints with documented lex-order semantics); the
bridge functions are responsible for assigning sensible values. The
field NAMES (`loop_index`, `vmfma_index`) are CMS-flavored — see §5 for
the open question on whether to rename them now or leave them.

### 3.3 Body labels — keep the four CMS labels

`BODY_LABEL_ML_PREV`, `BODY_LABEL_ML`, `BODY_LABEL_NGL`,
`BODY_LABEL_NLL` are baked into:

  * `_BODY_BUILD_ORDER` (CMSValidator.py:859)
  * `BODY_LABEL_TO_LOOP_INDEX` (ScheduleCapture.py:385)
  * The `ref_ids != subj_ids` identity-coverage check
    (CMSValidator.py:2403, indirectly via `_DATA_FLOW_KINDS`)
  * `validate_middle_pack_pair_interleaving` (uses `_all_nodes_in_order`
    which iterates `_BODY_BUILD_ORDER`)

An asm-source Timeline can legally populate fewer than four bodies — the
graph builder already handles that case (`if 0 not in by_cp: continue`,
CMSValidator.py:941; `if label not in captures: continue`,
CMSValidator.py:965). So an asm Timeline with only `{"ML": [...]}` works
out of the box.

The bridge from asm is responsible for choosing which label to use for
the single-body case. Recommendation: use `BODY_LABEL_ML` for any asm
input that doesn't structurally distinguish prologue/main/cleanup, and
extend the bridge to discriminate later if needed. No Timeline-side
work required.

### 3.4 Per-body `name_to_idx` — keep as a sibling map

The cleanest place is to attach it to the body container:

```python
@dataclass
class TimelineBody:
    events: List[TimelineEvent]
    # CMS-specific symbolic-vgpr-name → numeric-base map (rocm-libraries-bb34).
    # Populated by `cms_capture_to_timeline` from the writer's RegSet
    # directives. The asm bridge leaves this empty (an asm dump has
    # already-resolved numeric registers; symbolic-name resolution is
    # not a concern there). See §5 for the source-agnostic discussion.
    name_to_idx: dict = field(default_factory=dict)
```

And then `Timeline.events_by_body: Dict[str, TimelineBody]` instead of
`Dict[str, List[TimelineEvent]]`.

Tradeoff: a `TimelineBody` adds one indirection
(`timeline.events_by_body[label].events`) versus inlining `events` and
`name_to_idx` as parallel dicts on `Timeline`. The body wrapper wins
because the graph builder also needs to attach the `_graph_nodes`
sidecar to **something** per body, and a `TimelineBody` is the natural
home (vs polluting `Timeline` with another parallel dict). The current
producer side stashes `_graph_nodes` directly on `LoopBodyCapture`
(CMSValidator.py:1003); replacing `LoopBodyCapture` with `TimelineBody`
preserves that pattern exactly.

---

## 4. Worked example — `cms_capture_to_timeline` for `test_vswap_pair_reorder_detected`

The fixture lives at
`projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_dataflow_graph_register_gaps.py:3220`.

It is a good worked example because:

  * minimal — just two `VSwapB32` instructions in a single body;
  * exercises edge formation directly (the test body asserts an
    `OrderInvertedFailure` from `compare_graphs`);
  * only uses the canonical `_tag` + `make_capture` + `_wrap` helpers,
    so the FourPartCapture construction is concrete and short;
  * just landed (post-wx9.3) so the shape is current.

### 4.1 Current FourPartCapture construction (verbatim)

```python
sw1 = VSwapB32(dst=vgpr(0, 1), src=vgpr(1, 1))
sw2 = VSwapB32(dst=vgpr(1, 1), src=vgpr(2, 1))
ref_cap = make_capture(BODY_LABEL_ML, [
    _tag(sw1, category="PackA0", mfma_index=0, sequence=0),
    _tag(sw2, category="PackA0", mfma_index=0, sequence=1),
])
# ... and _wrap(ref_cap) builds the FourPartCapture:
FourPartCapture(
    main_loop={0: ref_cap},
    main_loop_prev={0: _filler(BODY_LABEL_ML_PREV)},  # one filler MFMA
    n_gl={0: _filler(BODY_LABEL_NGL)},                # one filler MFMA
    n_ll={0: _filler(BODY_LABEL_NLL)},                # one filler MFMA
    num_mfma=1,            # write-only — never read by the validator
    num_codepaths=1,       # write-only — never read by the validator
    source="cms",          # write-only — never read by the validator
    arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
)
```

`_tag` builds a `TaggedInstruction(wrapped=WrappedInstruction(inst),
category=..., slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
mfma_index=..., sequence=...))` and `make_capture` calls `_populate_wrapper`
on each `wrapped` so `wrapped.reads`/`wrapped.writes`/`wrapped.read_slots`/
`wrapped.write_slots` are populated.

For the two VSwaps the rule registry's `_GenericALURule` (or the
appropriate `_VSwapRule`) yields:

| event | reads | writes | read_slots | write_slots |
|---|---|---|---|---|
| sw1 (`VSwap(v0, v1)`) | `(vgpr(0,1), vgpr(1,1))` | `(vgpr(0,1), vgpr(1,1))` | `(0, 1)` | `(0, 1)` |
| sw2 (`VSwap(v1, v2)`) | `(vgpr(1,1), vgpr(2,1))` | `(vgpr(1,1), vgpr(2,1))` | `(0, 1)` | `(0, 1)` |

(VSwap is symmetric R+W; both operands appear on both sides at the same
slot — `dst` at slot 0, `src` at slot 1. This is the case wx9.3 phase 3
fixed.)

### 4.2 Worked-example `cms_capture_to_timeline(fp_cap) -> Timeline`

```python
# ILLUSTRATIVE — not implementation. The actual function lives in
# rocm-libraries-xe5; this block shows the field-by-field mapping.
def cms_capture_to_timeline(fp_cap: FourPartCapture) -> Timeline:
    bodies: Dict[str, TimelineBody] = {}

    body_sources = (
        (BODY_LABEL_ML_PREV, fp_cap.main_loop_prev),
        (BODY_LABEL_ML,      fp_cap.main_loop),
        (BODY_LABEL_NGL,     fp_cap.n_gl),
        (BODY_LABEL_NLL,     fp_cap.n_ll),
    )
    for label, by_cp in body_sources:
        if 0 not in by_cp:
            continue                                        # absent body — see CMSValidator.py:941
        loop_body = by_cp[0]                                # LoopBodyCapture
        events: List[TimelineEvent] = []
        for ti in loop_body.instructions:
            wrapped = ti.wrapped
            events.append(TimelineEvent(
                rocisa_inst = wrapped.rocisa_inst,          # CMSValidator.py:835
                reads       = tuple(wrapped.reads),         # CMSValidator.py:1063
                writes      = tuple(wrapped.writes),        # CMSValidator.py:1090
                read_slots  = tuple(wrapped.read_slots),    # CMSValidator.py:1060
                write_slots = tuple(wrapped.write_slots),   # CMSValidator.py:1087
                category    = ti.category,                  # CMSValidator.py:836
                position    = make_position(label, ti.slot),# CMSValidator.py:837
                body_label  = label,                        # implicit today; lifted onto the event
            ))
        bodies[label] = TimelineBody(
            events       = events,
            name_to_idx  = dict(loop_body.name_to_idx),     # CMSValidator.py:1054
        )

    return Timeline(
        events_by_body       = bodies,
        arch_profile         = getattr(fp_cap, "arch_profile", None),         # CMSValidator.py:956
        num_mfma_per_subiter = getattr(fp_cap, "num_mfma_per_subiter", 0) or 0,# CMSValidator.py:951
    )
```

### 4.3 Field-by-field accounting for `test_vswap_pair_reorder_detected`

Applying the bridge above to `ref_cap` produces:

```python
Timeline(
    events_by_body = {
        "ML-1": TimelineBody(
            events = [
                TimelineEvent(rocisa_inst=<MFMAInstruction filler>,
                              reads=(vgpr(204,2), vgpr(208,2)),
                              writes=(vgpr(200,4),),
                              read_slots=(0, 1), write_slots=(0,),
                              category="MFMA",
                              position=SchedulePosition(0, 0, 0),
                              body_label="ML-1"),
            ],
            name_to_idx = {},
        ),
        "ML": TimelineBody(
            events = [
                TimelineEvent(rocisa_inst=sw1,
                              reads=(vgpr(0,1), vgpr(1,1)),
                              writes=(vgpr(0,1), vgpr(1,1)),
                              read_slots=(0, 1), write_slots=(0, 1),
                              category="PackA0",
                              position=SchedulePosition(1, 0, 0),
                              body_label="ML"),
                TimelineEvent(rocisa_inst=sw2,
                              reads=(vgpr(1,1), vgpr(2,1)),
                              writes=(vgpr(1,1), vgpr(2,1)),
                              read_slots=(0, 1), write_slots=(0, 1),
                              category="PackA0",
                              position=SchedulePosition(1, 0, 1),
                              body_label="ML"),
            ],
            name_to_idx = {},
        ),
        "NGL": TimelineBody(events=[<filler MFMA>], name_to_idx={}),
        "NLL": TimelineBody(events=[<filler MFMA>], name_to_idx={}),
    },
    arch_profile         = _DEFAULT_CDNA4_ARCH_PROFILE,
    num_mfma_per_subiter = 0,
)
```

Every consumer field from §1 is accounted for. The `FourPartCapture`
fields **not** in the output (`num_mfma`, `num_codepaths`, `source`) are
exactly the fields §1 confirmed are never read by the validator.

---

## 5. Fields that don't map cleanly — open issues

### 5.1 `name_to_idx` is CMS-specific — keep it on `TimelineBody` for now

This map exists because the CMS kernel writer emits both symbolic
(`vgpr("ValuA_X0_I0", 4)`) and numeric (`vgpr(76, 4)`) references to the
same physical register and the byte-key resolver needs to collapse
them. Asm-source Timelines see already-resolved numeric registers, so
they leave `name_to_idx` empty.

  * The field is per-body (each `FourPartCapture` body has its own map)
    so cross-body name collisions don't bleed.
  * The consumer reads it via `getattr(body_capture, "name_to_idx", None)`
    (CMSValidator.py:1054), which already tolerates absence.

**Decision: keep `name_to_idx` on `TimelineBody` as an optional dict.**
It's a CMS-source artifact, but its consumer-side use is generic
("symbolic-name → numeric-base resolution for byte-key lookup"). An asm
bridge that later wants to model symbolic asm operands can populate it
the same way; until then asm bodies leave it empty and the resolver
falls through to the legacy symbolic-keying path. No source-aware
discrimination needed at the Timeline boundary.

### 5.2 `cms_node_label`'s per-category-stream `[N]` index is CMS-specific

`cms_node_label` (CMSValidator.py:2283) computes `LRA0[3]` by counting
how many same-category `TaggedInstruction`s precede this one in the
body. The label rendering is CMS-flavored; the asm-source equivalent
would render as `ds_load_b128 @ stream_pos=42` (per the docstring on
`FailureNodeLabel`).

**Decision: out of scope for this spike.** Label rendering is already
factored behind `FailureNodeLabel` and label-provider callbacks (see
sibling bead `rocm-libraries-3dy` / 5gd.B.1: source-aware
`tagged_inst` label dispatch). The Timeline interface itself doesn't
need to know about labels; `cms_node_label` consumes
`body_capture.instructions` (in our spike: `body.events`) and
`tagged_inst.category` (in our spike: `event.category`) — both
fields already on the `TimelineEvent`/`TimelineBody` shape. The
function as-written can be ported to walk `TimelineBody.events`
unchanged once we attach the equivalent of `node.tagged_inst` (i.e.
the originating `TimelineEvent`) to `GraphNode`.

The implementation note for the consuming bead: rename
`GraphNode.tagged_inst` to `GraphNode.event` (or keep the field name
and store a `TimelineEvent` in it). This is a Part-A.4 detail
(`rocm-libraries-iig`), not a Part-A.1 spike concern.

### 5.3 `SchedulePosition` field NAMES are CMS-flavored

`loop_index` reads naturally for a CMS capture (loop body index 0–3)
but reads oddly for an asm Timeline (where it might encode "asm-section
index" or be uniformly 0). Same for `vmfma_index` (asm Timelines have
no MFMA-index concept).

**Open question for the user:** rename now (e.g. `body_index`,
`stream_index`, `tie_break_index`) or accept the CMS-flavored names as
the price of zero-churn migration?

  * Renaming now means touching every consumer that reads
    `position.vmfma_index` or `position.loop_index` (audit count: ~10
    in CMSValidator.py + the renderer in `cms_node_label`).
  * Accepting them means asm-side users see CMS jargon in field names
    that don't quite mean what the names imply.

I lean toward **rename now** because the parent bead's "no
backwards-compat shims" directive applies symmetrically: if we leave
the names CMS-flavored we are baking a one-source assumption into the
generic interface. Renaming is a rote search-and-replace covering the
file:line list in §1.5 plus the rendering in `cms_node_label`. But
this is the user's call — the field names are visible across the
sibling beads in the chain (`rocm-libraries-iig`, `-3dy`, `-7qm`).

### 5.4 The graph builder's `body._graph_nodes` write-back

`build_dataflow_graph` writes `body._graph_nodes` onto the body
(CMSValidator.py:1003) and `_all_nodes_in_order` reads it
(CMSValidator.py:545). This is a sidecar attached at runtime to a
producer-owned object; it's not data the Timeline carries IN.

**Decision: same pattern, attach to `TimelineBody`.** Because
`TimelineBody` is a dataclass we own (not an opaque rocisa wrapper),
the `_graph_nodes` sidecar can live as a regular optional field on
`TimelineBody`:

```python
@dataclass
class TimelineBody:
    events: List[TimelineEvent]
    name_to_idx: dict = field(default_factory=dict)
    # Populated by `build_dataflow_graph` Phase 1; consumed by
    # `_all_nodes_in_order`, `waits_in_window`, `barriers_in_window`.
    # Always None on inputs to `build_dataflow_graph`; written exactly
    # once during graph construction. (Could be promoted to a
    # graph-builder-local Dict[body_label, List[GraphNode]] in a future
    # cleanup, but that's out of scope here.)
    graph_nodes: Optional[List[object]] = None
```

This is mechanically equivalent to the current behavior. No bridge
helper needed.

### 5.5 `WrappedInstruction.rocisa_inst` vs `TimelineEvent.rocisa_inst`

`WrappedInstruction.rocisa_inst` is a property; `__getattr__` forwards
arbitrary attribute access to the underlying rocisa instance. The
validator currently does `node.tagged_inst.wrapped.rocisa_inst` to
reach it (CMSValidator.py:835), and `_swait_drains` does
`node.rocisa_inst` directly (which works because `_make_node` sets
`GraphNode.rocisa_inst = inst`).

For `TimelineEvent`:

  * The consumer never `__getattr__`-forwards through the wrapper —
    every consumer site reads either `wrapped.rocisa_inst` (always) or
    one of the four explicit fields `wrapped.reads`, `wrapped.writes`,
    `wrapped.read_slots`, `wrapped.write_slots`. Confirmed by §1.4.
  * So `TimelineEvent` can store `rocisa_inst` directly and the four
    other fields as dataclass fields. The `WrappedInstruction` proxy
    layer disappears at the consumer boundary.

**Decision: drop `WrappedInstruction` from the consumer-facing shape.**
The producer side (`LoopBodyCaptureBuilder`) can keep using
`WrappedInstruction` as its in-memory carrier during capture — the
bridge function copies the four fields out and discards the wrapper.
This is the only field that "doesn't map cleanly" in the strict sense
(the type structurally collapses), and the cleanup is straight
forward.

### 5.6 Summary table

| Field | Source | Decision | Where the cleanup lives |
|---|---|---|---|
| `name_to_idx` | CMS-specific | Keep on `TimelineBody` as optional dict; asm leaves empty. | Bridge bead `rocm-libraries-xe5` populates; bridge bead `rocm-libraries-x6s` leaves empty. |
| Per-category-stream `[N]` index for `cms_node_label` | CMS-specific | Out of scope; label rendering is sibling bead `rocm-libraries-3dy`. The TimelineEvent fields the labeler needs (`category`, `body.events`) already exist. | Sibling bead `rocm-libraries-3dy`. |
| `SchedulePosition` field names (`loop_index`, `vmfma_index`) | CMS-flavored | **OPEN — user decision.** Rename now (recommended) or accept CMS names as legacy. | If rename: this spike doc plus all consumer sites in §1.5 (~10 lines in CMSValidator.py). If keep: noted in this doc; no code change. |
| `body._graph_nodes` write-back | Graph-builder sidecar (not input) | Promote to typed `Optional[List[GraphNode]]` field on `TimelineBody`. Mechanically equivalent. | Sibling bead `rocm-libraries-iig` (validator entry routing). |
| `WrappedInstruction` proxy layer | Producer-side carrier | Collapse into `TimelineEvent` direct fields. Producer keeps it; consumer stops seeing it. | Bridge bead `rocm-libraries-xe5`. |
| `FourPartCapture.num_mfma`, `num_codepaths`, `source` | Producer-side metadata | Drop — never read by the validator. | This spike (catalogued in §1.1); confirmed by greps over Components/. |
| `SlotKey.subiter`, `SlotKey.slot_kind` | Producer-side bucketing | Drop from the consumer-facing shape — never read by `make_position` or anything downstream of it. | This spike (catalogued in §1.6); the bridge function never copies them. |

---

## 6. The one open question to resolve before Part A (`rocm-libraries-3g4`) starts

**Should `SchedulePosition`'s field names (`loop_index`, `vmfma_index`,
`sub_index`) be renamed to source-agnostic ones (`body_index`,
`stream_index`, `tie_break_index`) before A.4 routes the validator
entry through `Timeline`?**

  * If **yes**: rename happens in `rocm-libraries-iig` alongside the
    entry-point rewiring. Affects ~10 consumer sites in
    `CMSValidator.py` (catalogued in §1.5) plus the rendering string
    `@ idx={vmfma_index}` in `cms_node_label`. Mechanical, single
    commit.
  * If **no**: the names persist and asm-source Timelines have to live
    with CMS jargon in their position field. The asm bridge bead
    (`rocm-libraries-x6s`) has to document the mapping in its
    docstring (`stream_index → vmfma_index`).

I recommend **yes**, on the grounds that the parent bead's directive is
"no fallback paths, no parallel APIs": baking CMS-flavored names into
the generic interface is a soft form of leaving the old shape in
place. But the user owns the call.

Everything else in §5 is cleanly resolvable in the bridge or sibling
beads without further spike work.

---

## 7. What's left for the implementation beads

For the implementer reading just this doc plus the parent bead
description, the to-do list across the chain is:

  * `rocm-libraries-xe5` (5gd.A.2 — `cms_capture_to_timeline`):
    implement the bridge per §4.2. No new fields beyond the §2 / §3
    catalogue.
  * `rocm-libraries-x6s` (5gd.A.3 — `assembly_to_timeline`): implement
    the asm bridge. Populate the same `TimelineEvent` fields,
    leaving `name_to_idx` empty. Sub-bead `rocm-libraries-7qm` covers
    the rocisa-instruction → TimelineEvent (reads/writes/edges)
    mapping inside this bridge.
  * `rocm-libraries-iig` (5gd.A.4 — route validator entry through
    Timeline): change `build_dataflow_graph(four_part_capture)` to
    `build_dataflow_graph(timeline: Timeline)`, switching every
    `four_part_capture.main_loop[0]` access to
    `timeline.events_by_body["ML"].events`. Same for the other three
    body labels. The graph builder's `_make_node` (CMSValidator.py:830)
    takes a `TimelineEvent` instead of a `TaggedInstruction`; the field
    accesses change shape but not semantics.
  * `rocm-libraries-3dy` (5gd.B.1 — source-aware label dispatch):
    factor `cms_node_label` behind a label-provider hook so an
    asm-source Timeline supplies its own labeler (`asm_node_label`).
    The Timeline interface itself doesn't change; this is the rendering
    boundary.

No part of the above requires new fields on `TimelineEvent` or
`TimelineBody` beyond what §2 / §3 already proposes.
