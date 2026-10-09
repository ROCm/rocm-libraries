# Refresh S00-S07 emit results after develop changes

## Status

Accepted

## Context

The S00-S07 tests generate assembly on the CPU from small designed YAML files
and save each kernel's content-derived basename and emitter return code. After
rebasing onto current `develop` and rebuilding `rocisa` from the same checkout,
20 saved results no longer matched their generated basenames.

The previous basenames were also not reproducible from the pre-rebase stack tip
with a fresh in-tree `rocisa` build. For all 20 affected nodes, the current run
preserves the number of kernels and the multiset of emitter return codes. Only
the content-derived names change.

## Decision

Re-record only the 20 failing S00-S07 saved-result nodes against current
`develop` and a freshly built in-tree `rocisa`. Keep each designed YAML input,
kernel count, and emitter result unchanged.

## Consequences

The saved results now reproduce with the source and `rocisa` implementation in
this branch. Future name changes require the same count and return-code audit;
the saved basenames alone are not evidence that emitted assembly is correct.
