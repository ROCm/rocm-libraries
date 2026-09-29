# Downstream JIT design notes

These contributor notes describe just-in-time (JIT) generation on
`users/jolabega/downstream-hipblaslt-jit-develop`. The [JIT guide](../JIT.md) is
the plan of record: it separates current behavior from the approved target
design and carries the roadmap. The notes below develop specific parts of that
design and record source evidence.

- [Discussion roadmap](confluence-roadmap-draft.md): the September 29 design
  revision, ready to copy into Confluence. It presents the target design:
  heuristic-driven generation controlled by `HIPBLASLT_JIT`, the Jit backend
  interface with Predictor and TuningKnowledge, in-process code-object
  construction through AMD comgr, the persistent JIT solution library, and
  removal of the explicit entry points from the public application programming
  interface (API). It records the decisions settled since the September 28
  draft and the ones still open. Because Confluence does not render Mermaid, it
  shows the diagram as text and as an edge table. This revision has not been
  published to Confluence.
- [KFA producer convergence](kfa-producer-convergence.md): KernelFromAnywhere
  (KFA) contract discovery, producer gaps, concrete type reuse, rocRoller/source
  evidence, and selection constraints. Complete producer metadata precedes proof
  and shared dispatch; the earlier external-ingestion-first recommendation
  remains superseded. The September 29 update revises its selection, public API
  and rocRoller sections for the target design.
- [Host timing and progress](timing-host-plan.md): independent opt-in
  `HIPBLASLT_JIT_DEBUG` categories, provider/process boundaries, report transport
  and live progress observation.
- [Python timing and progress](timing-python-plan.md): compiler/build-stage
  boundaries, failure accounting and the coordinated child protocol.

The investigations identify inspected revisions; earlier discovery and timing
line numbers remain historical anchors. The timing plans predate the target
design. Roadmap step 1 makes the headers they cite internal, and step 3 moves
assembly, linking and helper compilation from Python-launched tools into
hipBLASLt. Re-derive their compile-stage boundaries when timing work begins.

These notes do not implement the target design, KFA convergence or the timing
features; the [roadmap](../JIT.md#roadmap) records implementation status. The external Confluence reference is a format/tone example, not an
identified publication destination; see the discussion draft for its access
limitation.

Start with the [downstream handoff](../../../SESSION_HANDOFF.md) for branch
preservation, validation limits and the existing build environment. The
[JIT guide](../JIT.md), [TensileLite backend guide](../JIT_TENSILELITE.md) and
[single-solution guide](../tensilelite/SINGLE_SOLUTION.md) remain the
implementation guides.
