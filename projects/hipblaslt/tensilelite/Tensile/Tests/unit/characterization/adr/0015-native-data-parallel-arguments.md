# ADR 0015: Give generated DataParallel kernels a native scheduling payload

Status: Accepted
Supersedes: the retained argument-layout clause in [ADR 0014](0014-canonical-persistent-policy-names.md)

## Context

Persistent DataParallel kernels process full K tiles. The legacy six-word
scheduling payload describes StreamK iteration partitions and requires an
iteration cursor plus magic division to recover the next tile. Those fields
are unnecessary for full-tile assignment. Prebuilt kernels still depend on
their recorded payload, argument offsets, and explicit names.

## Decision

Generated DataParallel/StaticGrid kernels use `PersistentLoopArgsVersion=1`
with outer `KernArgsVersion=3`. Their scheduling payload is two adjacent
32-bit words, `ItersPerTile` and `PersistentGrid`. Assignment advances a tile
cursor; full K processing retains the physical-K and alpha-zero exits.

Prebuilt kernels without an explicit scheduling version remain version zero.
Their six-word payload and outer versions 0–3 remain supported. Explicit
native descriptors must match the two-word contract. Selector overrides
rederive generated layouts and reject incompatible explicit prebuilt layouts.

## Consequences

Native DP argument offsets and register requirements intentionally differ from
the legacy layout. Signature tests, complete host-buffer comparisons, emitted
instruction checks, and numerical zero-K/zero-alpha/PAP cases validate those
differences. StreamK and ordinary kernels keep their prior payloads.

The naming decision in ADR 0014 continues to apply. Any characterization
expectation affected by this layout change must be classified and updated at
its exact node, then pass twice without recording. Basename-only changes on
stable architectures do not establish numerical correctness.
