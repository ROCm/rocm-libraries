# Refresh S08-S11 emit results after develop changes

## Status

Accepted

## Context

The S08-S11 tests generate assembly on the CPU from small designed YAML files
and save each kernel's content-derived basename and emitter return code. After
rebasing onto current `develop` and rebuilding `rocisa` from the same checkout,
11 saved results no longer matched their generated basenames.

Each affected node produces the same number of kernels as before, and each
node preserves its multiset of emitter return codes. The changed values are
limited to content-derived basenames.

## Decision

Re-record only the 11 failing S08-S11 saved-result nodes against current
`develop` and a freshly built in-tree `rocisa`. Keep every designed YAML input,
kernel count, and emitter result unchanged.

## Consequences

The saved results now reproduce with the source and `rocisa` implementation in
this branch. Future name changes require the same count and return-code audit
before their saved results can be updated.
