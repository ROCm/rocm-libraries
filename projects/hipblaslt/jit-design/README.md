# JIT design notes

These contributor notes develop specific parts of the just-in-time (JIT)
generation design. The [JIT guide](../JIT.md) is the plan of record: it
separates current behavior from the approved target design and carries the
roadmap.

- [KFA producer convergence](kfa-producer-convergence.md): KernelFromAnywhere
  (KFA) contract discovery, producer gaps, concrete type reuse, rocRoller, and
  selection constraints. Complete producer metadata precedes proof and shared
  dispatch.
- [Host timing and progress](timing-host-plan.md): independent opt-in
  `HIPBLASLT_JIT_DEBUG` categories, provider/process boundaries, report transport
  and live progress observation.
- [Python timing and progress](timing-python-plan.md): compiler/build-stage
  boundaries, failure accounting and the coordinated child protocol.

The notes cite source locations as `file:line`. Line numbers locate the named
functions approximately; search for the named symbol when they drift. The timing
plans describe the current build pipeline, in which Python launches the
assembler, linker and helper compiler. Roadmap step 3 moves that work into
hipBLASLt through AMD comgr, so re-derive the compile-stage boundaries when
timing work begins.

These notes are design material. KFA convergence and the timing features are
not implemented; the [roadmap](../JIT.md#roadmap) records implementation status.
The [JIT guide](../JIT.md), [TensileLite backend guide](../JIT_TENSILELITE.md)
and [single-solution guide](../tensilelite/SINGLE_SOLUTION.md) are the
implementation guides, and the [JIT test guide](../clients/tests/jit/README.md)
covers building and testing.
