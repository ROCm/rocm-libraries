# ADR 0037: LocalDotLayout added as a solution parameter

Status:  Accepted
Defect:  none — the parameter records an intended behavior, not a pinned bug

## Context

TensileLite 5 dropped `LocalDotLayout`, the solution parameter that arranges the elements of a
row in LDS so that N consecutive K elements land in one VGPR. Without it the packed-int8 path has
only the classic one-byte-per-register MAC, whose local write scatters four bytes over four
32-byte planes while its local read fetches four contiguous bytes. Only one byte in four
survives, and the GEMM result is silently wrong.

The branch re-adds the parameter so that `LocalDotLayout=4` can select the four-way signed 8-bit
dot product (`v_dot4c_i32_i8`), which is the path AMD's own shipped rocBLAS gfx1030 int8 kernel
uses. The parameter takes the values 1, 2, 4 and 8, and 1 — no interleaving — is the default, so
a logic file that does not set it is unaffected in behavior.

Adding a solution parameter is visible in the serialized data rather than only in the selection.
It appears in every solution's dictionary, in the logic contract that `LibraryIO` reads and
writes, and in the full solution name, which gains an `LDL1` token. The characterization goldens
pin all three, so 18 golden nodes across three files moved: twelve in
`LibraryIO/__snapshots__/test_logiccontract_char.ambr`, four in
`SolutionClass/__snapshots__/test_solution_class_char.ambr`, and two in
`ValidParameters/__snapshots__/test_builders_char.ambr`.

The branch was rebased onto upstream `develop` after the goldens were recorded. Upstream's
base adds a solution parameter of its own, `PostLoopStoreInNll`, so the solution key count
this branch starts from is one higher than the 354 the re-record was first made against.

## Decision

Re-record the 18 affected golden nodes one node at a time (`pytest <node-id> --snapshot-update`)
rather than regenerating the suite, and record the parameter's addition in the decision log. The
change is an intended behavior change, so no golden pins a defect and no `Defect:` link is
carried.

## Consequences

Every solution's full name changes on every architecture, because the full name lists the valid
parameters. The count of solution keys goes from 355 to 356. A user tuning file that stores a
solution name beside its index sees the new token.

The kernel identity does not change. `LocalDotLayout` is not in the required (min-naming)
parameter set, so no kernel basename or hash moves: the `_codegen` goldens that pin the emitted
assembly and the basename hashes are byte-identical, and the emitted code objects are unchanged.

Each affected golden node gains exactly one entry and nothing else moves, apart from the nodes
that record the solution's key count, which goes from 355 to 356. A future change to the
parameter's name, its default or its set of values re-breaks the same nodes; if the parameter is
ever removed, flip the goldens back and supersede this ADR.
