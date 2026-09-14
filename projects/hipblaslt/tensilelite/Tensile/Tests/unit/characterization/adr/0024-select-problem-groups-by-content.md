# Select problem groups by content

## Status

Accepted

## Context

ADR 0021 made the selected `BenchmarkProblems` index visible in each set-cover
test. Every case selected index zero. A numeric index still refers to a
different problem group when a shared YAML file inserts or reorders entries, so
it did not provide the stable selection promised by that decision.

## Decision

Compute a 12-hex-character SHA-256 fingerprint from the complete normalized
`BenchmarkProblems` entry. Set-cover tables record that fingerprint, and the
harness searches all entries for it. If the selected entry is removed or
changed, the test fails with a message that names the missing fingerprint and
configuration file.

Index selection remains available for small designed fixtures and callers that
intentionally address a list position.

## Consequences

Inserting or reordering unrelated problem groups no longer changes a set-cover
test's input. An intentional edit to the selected problem type or size group
requires updating its fingerprint. Identical duplicate entries share a
fingerprint, but selecting either is behaviorally equivalent.
