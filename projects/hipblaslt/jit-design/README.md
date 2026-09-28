# Downstream JIT design notes

These contributor notes describe just-in-time (JIT) generation on
`users/jolabega/downstream-hipblaslt-jit-develop`.
The current application programming interface (API) names are retained.
The notes distinguish implementation from proposed KernelFromAnywhere (KFA)
convergence, additional generators, fallback selection, caching, and diagnostics.

- [Discussion roadmap](confluence-roadmap-draft.md): the September 25 design draft,
  ready to copy into Confluence. It compares existing problem/solution reuse,
  TensileLite and rocRoller integration, a shared KFA consumer, and an optional
  zero-result fallback library. This revision has not been published to Confluence.
- [Host timing and progress](timing-host-plan.md): independent opt-in categories,
  provider/process boundaries, report transport and live progress observation.
- [Python timing and progress](timing-python-plan.md): compiler/build-stage
  boundaries, failure accounting and the coordinated child protocol.
- [KFA producer convergence](kfa-producer-convergence.md): contract discovery,
  producer gaps, concrete type reuse, rocRoller/source evidence, and selection
  constraints. Complete producer metadata precedes proof and shared dispatch;
  the earlier external-ingestion-first recommendation remains superseded.

The investigations identify inspected revisions; earlier discovery and timing
line numbers remain historical anchors. None of these design updates implements
the proposed convergence, fallback library, persistent cache, or timing features.
The external Confluence reference is a format/tone example, not an identified
publication destination. See the discussion draft for its access limitation.

Start with the [downstream handoff](../../../SESSION_HANDOFF.md) for branch
preservation, validation limits and the existing build environment. The
[versioned roadmap](../JIT_ROADMAP.md), [direct API guide](../JIT_TENSILELITE.md)
and [generic API guide](../JIT.md) remain the implementation guides.
