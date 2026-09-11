# Select config problem groups explicitly

## Status

Accepted

## Context

The config-driven harness always read `BenchmarkProblems[0]`. The set-cover
tables supplied only a YAML path and architecture, although 57 of their 75
files contain multiple problem groups. A reader could not tell which group was
measured, and inserting or reordering a group could silently redirect a test.

## Decision

Add a `problem_index` argument to the solution and emission helpers. Record an
explicit index in every set-cover case and include it in the pytest case name.
Reject indexes outside the selected YAML's problem list.

## Consequences

Each set-cover node names the exact input group it exercises. Adding another
group to a shared YAML no longer changes the selected behavior unless the test
table is deliberately updated.

The current harvest continues to use index zero for every case; this change
documents that scope rather than expanding the number of generated kernels.
