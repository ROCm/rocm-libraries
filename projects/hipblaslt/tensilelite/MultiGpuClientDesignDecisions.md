# TensileLite multi-GPU client design decisions

This is the living, committed record for the multi-GPU expansion of
`tensilelite-client`. It records only decisions confirmed in discussion, their
rationale, and their consequences. Factual source investigation is kept in
[`FusedA2AClientMultiGpuResearch.md`](FusedA2AClientMultiGpuResearch.md).

## Status

Design exploration is in progress. No product-code implementation has been
approved by this record yet.

## Decision MGPU-001: extend the existing client; do not retain a fused client

**Status:** Accepted on 2026-08-27

**Decision:** Evolve the existing `tensilelite-client` into the general
multi-GPU runner. A one-GPU invocation must remain the one-rank case of that
same client. The GEMM + A2A epilogue and a future A2A + GEMM prologue must be
operation modes within the shared client execution path. The bespoke
`FusedA2AClient.cpp` path is a migration target and must ultimately be
eliminated rather than retained as a second client architecture.

**Rationale:** The ordinary client currently owns one device, one stream, one
data-initialization lifecycle, and one reference-validation lifecycle. The
branch instead bypasses those paths with a special multi-GPU runner that
duplicates topology, resource ownership, kernel-argument packing, launch,
synchronization, and validation behavior. Retaining both paths would make new
collective directions, correct all-rank validation, and future maintenance
depend on two incompatible client models.

The existing Python-level parallel execution facility is not a substitute: it
only distributes independent benchmark problems to separate processes and
merges their CSV output; it does not represent ranks or execute a collective.

**Consequences:**

- The design must identify how to generalize existing client data preparation,
  timing, result reporting, and validation rather than recreate them in a
  feature-specific harness.
- Multi-GPU resources and collective-specific inputs/outputs must become
  explicit parts of the existing client's model.
- No public TensileLite or hipBLASLt production runtime API expansion is
  implied by this decision. That boundary remains a separate decision if it
  becomes necessary.
- Existing `FusedA2AClient.cpp` behavior is evidence and a migration source,
  not the target interface or ownership model.

**Evidence used:**

- [`FusedA2AClientMultiGpuResearch.md`](FusedA2AClientMultiGpuResearch.md)
  documents the current normal and fused execution paths with source anchors.
- PR #10925 describes `FusedA2AClient.cpp` as a special 4-GPU validation
  harness rather than an extension of the normal client flow.

**Next unresolved decision:** Define the supported topology and process scope
for the general multi-GPU client (local visible devices, rank mapping,
single-process versus multi-process, and node boundaries).
