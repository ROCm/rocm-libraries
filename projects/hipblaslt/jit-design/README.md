# JIT design notes

These contributor notes develop specific parts of the just-in-time (JIT)
generation design. The [JIT guide](../JIT.md) is the plan of record: it
separates current behavior from the approved target design and carries the
roadmap.

- [KFA producer convergence](kfa-producer-convergence.md): KernelFromAnywhere
  (KFA) contract discovery, producer gaps, concrete type reuse, and selection
  constraints. Complete producer metadata
  precedes proof and shared dispatch.

The notes cite source locations as `file:line`. Line numbers locate the named
functions approximately; search for the named symbol when they drift.

These notes are design material. KFA convergence is not implemented; the
[roadmap](../JIT.md#roadmap) records implementation status.
The [JIT guide](../JIT.md), [TensileLite backend guide](../JIT_TENSILELITE.md)
and [single-solution guide](../tensilelite/SINGLE_SOLUTION.md) are the
implementation guides, and the [JIT test guide](../clients/tests/jit/README.md)
covers building and testing.
