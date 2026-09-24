# Downstream JIT design notes

These contributor design notes are preserved with
`users/jolabega/downstream-hipblaslt-jit-develop`.
The current implementation and API names are retained. The notes describe
future work; copying them here does not implement timing, progress, KFA
convergence, persistent caching or a new publication workflow.

- [Discussion roadmap](confluence-roadmap-draft.md): the reusable Confluence
  draft at the end of upstream review. It has not been published to Confluence;
  PR references and review status describe that historical snapshot.
- [Host timing and progress](timing-host-plan.md): independent opt-in categories,
  provider/process boundaries, report transport and live progress observation.
- [Python timing and progress](timing-python-plan.md): compiler/build-stage
  boundaries, failure accounting and the coordinated child protocol.
- [KFA producer convergence](kfa-producer-convergence.md): source-grounded
  contract discovery, producer gaps and the selected producer-first sequence.
  Its earlier external-ingestion-first recommendation is explicitly superseded.

The timing/KFA investigations identify their inspected source revision and
use line numbers from that revision. Verify anchors before implementation.
The reusable KFA contract and implementation sequence are preserved here;
obsolete scratch review ledgers and footprint-audit bookkeeping are omitted.

Start with the [downstream handoff](../JIT_DOWNSTREAM_HANDOFF.md) for branch
preservation, validation limits and the existing build environment. The
[versioned roadmap](../JIT_ROADMAP.md), [direct API guide](../JIT_TENSILELITE.md)
and [generic API guide](../JIT.md) remain the implementation guides.
