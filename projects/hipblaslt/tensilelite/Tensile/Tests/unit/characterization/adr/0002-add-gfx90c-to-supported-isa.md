# ADR 0002: Add gfx90c to the supported ISA set

Status:  Accepted
Defect:  none — behavior is intended

## Context

TensileLite derives its valid architecture parameter from `SUPPORTED_ISA`.
Supporting gfx90c requires `IsaVersion(9, 0, 12)` to participate in the same
validation and code-generation paths as the existing supported architectures.
The characterization snapshot records the size of that architecture list, so
the durable support-surface change increases the recorded length from 23 to 24.

## Decision

Include `IsaVersion(9, 0, 12)` in `SUPPORTED_ISA` and update only the
`test_valid_parameters_structure` snapshot node that summarizes the list.

## Consequences

gfx90c is accepted by the normal TensileLite architecture validators and the
valid-parameter snapshot deliberately records the expanded support surface.
Removing or replacing gfx90c support must update the focused snapshot and
supersede this decision.
