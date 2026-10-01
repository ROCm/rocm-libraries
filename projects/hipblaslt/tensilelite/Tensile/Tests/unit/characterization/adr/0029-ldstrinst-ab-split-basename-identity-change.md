# ADR 0029: Accept basename identity changes from the LDSTrInstA/LDSTrInstB split

Status:  Accepted
Defect:  none — behavior is intended

## Context

This PR replaces the single global `LDSTrInst` kernel parameter with
independent per-tensor `LDSTrInstA`/`LDSTrInstB` overrides (`LDSTrInst` is
kept as the shared default both resolve from when left on `-1`). Both fields
are now part of every kernel's derived state and therefore participate in the
solution identity hash used to build each kernel's `basename`.

Several `_codegen` characterization goldens record exact `basename` strings,
including for stable architectures (gfx90a, gfx942) and other targets
(gfx950, gfx1250) that have no LDSTr hardware at all — `isLDSTrEnabled()`
returns `False` unconditionally on those ISAs, so `enableLDSTrA`/
`enableLDSTrB` (and therefore the generated assembly and `err` code) are
completely unaffected. Only the *name* changes, because the derived-state
dict now carries two keys (`LDSTrInstA`, `LDSTrInstB`) instead of one
(`LDSTrInst`) feeding the naming hash. Without this ADR, a reviewer seeing a
stable-architecture golden diff limited to a `basename` hash could reasonably
mistake it for an unintended codegen regression.

## Decision

Accept the `basename`-only golden changes across the affected snapshots
(`test_seed_gfx90a_*`, `test_seed_gfx942_*`, `test_seed_gfx950_*`,
`test_r3_*`, `test_r4_*`, and the gfx1250 StreamK cluster-multicast goldens)
as an expected consequence of adding `LDSTrInstA`/`LDSTrInstB` to kernel
derived state. Every affected node's `err` field (and, where recorded, its
assembly content) is unchanged — the diff is the hash text only.

## Consequences

A future golden diff that touches only `basename` hash text (no `err` or
assembly content change) on an architecture with no LDSTr hardware is not,
by itself, evidence of a behavior regression; verify the non-hash fields
first. If a later change removes `LDSTrInstA`/`LDSTrInstB` from the naming
hash's input set (e.g. by excluding parameters that resolve to the same
ineffective value on a given ISA), these basenames would need to be
regenerated again and this ADR superseded.
