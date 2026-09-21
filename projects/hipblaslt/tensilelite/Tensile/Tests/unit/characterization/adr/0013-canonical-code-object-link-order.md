# ADR 0013: Canonicalize code-object linker input order

Status:  Accepted
Defect:  [ROCM-28430](https://amd-hub.atlassian.net/browse/ROCM-28430)

## Context
Code-object inputs come from two paths: unnamed groups retain kernel iteration
order in a list, while explicit groups accumulate paths in an unordered set.
Fixing only the set-backed path would remove its hash-seed instability but
leave the two paths with different ordering rules.

## Decision
Sort object-file paths immediately before every linker invocation, establishing
one lexicographic order for both default and explicitly grouped code objects.

## Consequences
Identical inputs now produce a stable code-object layout and build ID. Default
groups also change from kernel iteration order to path order; this can change
physical kernel placement without changing the included kernels or their
assembled instructions. Any future ordering change must preserve a stable total
order or supersede this ADR.
