# ADR 0039: Enable gfx942 chip-ID selection without retagging generic logic

Status: Accepted
Defect: none - intended behavior change; tracked by AIHPBLAS-5961

## Context

gfx942 logic previously ignored PCI chip IDs. Enabling device-specific selection
for `0x74a5` introduces the directed fallback `0x74a5 -> 0x74a1`.
The saved `test_arch_source_default_partition[gfx942]` result derives its source
set from the fallback registry, so `id=74a5` moves from default to source.
This changes folder validation, not the architecture's supported chip-ID set.

## Decision

Accept that single partition change. Permit `gfx942_id74a5` logic while keeping
legacy `Device 0049` and `Device 0050` declarations chip-agnostic outside chip-ID
folders. Generate an `_ID` library-name suffix only when its hardware predicate
actually contains a chip-ID discriminator.

## Consequences

Existing generic gfx942 logic retains its hardware predicate and library name.
Device-specific rows precede CU-specific and generic rows. Fallback matching
does not make another chip's Equality solution eligible.
This PR reserves an empty device-specific Equality directory; tuned YAMLs will
arrive in a separate stacked PR. No tuned payload or performance gain is implied
by this saved-result update.
