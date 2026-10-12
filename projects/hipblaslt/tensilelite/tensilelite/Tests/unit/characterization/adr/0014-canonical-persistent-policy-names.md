# ADR 0014: Name generated kernels with the persistent execution policy

Status: Accepted
Defect: none — behavior is intended

## Context

Persistent execution now separates how a kernel processes a matrix tile from
how it receives work. `TileProcessingStrategy` and `WorkAssignment` replace the
numeric `StreamK` selector; shared mapping and queue controls also have new
names. Derived solutions and serialized defaults expose this policy explicitly.

Generated kernel names include those fields. Their shortened filename hashes
therefore change, including ordinary kernels whose strategy is `None`. The
saved name and schema expectations still describe the previous fields.

## Decision

Accept the explicit policy in generated names and solution schemas. Update
only characterization nodes whose differences follow from this migration;
preserve explicit names attached to prebuilt kernels. Retain the existing six-word scheduling payload and its version-zero
argument layout for generated and prebuilt kernels.

## Consequences

The updated code-generation expectations retain kernel counts and emission
return codes; their intended difference is the generated basename. Schema
expectations change only policy fields and the resulting key counts. These
checks do not establish numerical equivalence or validate unsupported GPU
targets. Each updated node must pass twice without recording, and unrelated
emission failures require diagnosis before any expectation changes.

Name-based ordering does not change the covered fixture membership here. The
updated capped logic fixtures contain exactly the number of solutions emitted
by their cap. Per-architecture logic suites have no cap. Config suites limit
fork permutations before deriving solutions, and each solution produces one
assembly kernel, so their later cap cannot discard a different representative
when names change.

A future naming or policy-schema change must explain its effect on generated
kernel identity and supersede this decision when the rule changes.
