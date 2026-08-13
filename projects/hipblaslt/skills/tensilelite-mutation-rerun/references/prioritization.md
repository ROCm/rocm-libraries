# Prioritizing mutation work

Choose the next mutation target from measured, reviewable evidence. The goal is
not a universal score; it is a defensible next-work decision whose inputs and
rationale remain understandable after the campaign context is gone.

## 1. Fix the candidate universe

List candidate source modules before comparing metrics. For each candidate,
record one status:

- `completed`: mutation run and triage certified for the pinned source/config;
- `ready`: covering set and environment are ready for a mutation slice;
- `coverage-gap`: tests do not yet meet the target-module threshold;
- `deferred`: a named dependency, cost, or design decision blocks work;
- `out-of-scope`: generated, third-party, data-only, or otherwise excluded with
  rationale; or
- `pending-evidence`: required metrics or provenance are missing.

Do not infer candidate modules from characterization directory names alone. A
directory may cover multiple modules, and one module may be covered by multiple
directories. Map source modules to explicit test selections.

Freeze the candidate list for one comparison. Adding or removing candidates can
change normalized ranks even when every raw metric is unchanged.

## 2. Prefer direct mutation and coverage evidence

Collect, in priority order:

1. Exact target-module line/branch coverage and uncovered-line count.
2. Prior non-killed counts by native status.
3. Strictly verified residual survivors and no-test pressure.
4. Inconclusive/timeout/segfault pressure requiring environment work.
5. Deterministic covering-set availability.
6. Cost: baseline duration, mutant count, and estimated rerun time.

Record source SHA, tool versions, configuration, test selection, command, and
artifact path for every measured value.

Do not compare mutation counts from incompatible mutmut versions, target sets,
pragmas, or covered-lines settings.

## 3. Use proxy metrics only as context

Complexity, LOC, import coupling, and recent source churn can help break ties,
but they are not substitutes for measured mutation/coverage evidence.

If used, define them precisely:

- LOC: physical, logical, or executable lines;
- complexity: tool/version and aggregation rule;
- coupling: parsed import graph scope, excluding or including tests explicitly;
- churn: time window and source of commit history.

Textual substring counts across Python files are not import-graph in-degree and
must not be labeled a floor or authoritative coupling metric.

## 4. Do not fake a composite score

Do not compute a score when required inputs are missing. Mark the value
`PENDING` and choose work through explicit review instead.

Before adopting a weighted score, require a committed decision record that
defines:

- why each metric predicts mutation-test value;
- metric units and valid ranges;
- fixed candidate universe;
- normalization baseline;
- weights and sensitivity analysis;
- treatment of missing/inconclusive inputs; and
- when the formula is versioned or retired.

Scores normalized over different candidate sets are not directly comparable.
Never retain hardcoded dispositions from a private or missing plan as if they
were computed output.

## 5. Apply a reviewable decision order

Use this default decision sequence:

1. Do not rerun a `completed` target unless source/config changed or regression
   evidence requires re-audit.
2. Address `coverage-gap` targets by adding deterministic covering tests before
   spending time on mutmut.
3. Resolve environment-driven inconclusive/timeout pressure before interpreting
   mutation score.
4. Among `ready` targets, prefer high verified survivor/no-test pressure and
   high product criticality with a manageable run cost.
5. Use complexity/coupling/churn only to break evidence-based ties.
6. Keep deferred and out-of-scope reasons explicit; never silently omit them.

Product criticality is a human judgment. Record why the module matters to
shipping behavior, validation, code generation, or frequently changed paths.

## 6. Decision record

For the selected module, write:

```text
candidate universe and source SHA
selected module
current status
covering-set artifact and target coverage
prior mutation/triage evidence
estimated cost
proxy metrics used, with definitions
why this candidate outranks the nearest alternatives
known blockers and required CI/device coverage
reviewer/owner decision
```

Store the decision with the campaign handoff or tracked issue. Do not create a
history directory that is neither committed nor consumed.

## 7. Re-evaluate

Rebuild the candidate evidence when:

- the base/source SHA changes materially;
- target or test selections change;
- mutmut/coverage versions or settings change;
- a target is completed;
- a new coverage gap or regression appears; or
- run-cost measurements invalidate the plan.

Preserve old receipts as historical evidence, but do not assert stale counts or
dispositions as current truth.

## 8. Future automation bar

Implement automated ranking only when there is a real recurring consumer and:

- all candidates map to source modules;
- all required metrics have validated schemas and provenance;
- missing/invalid ranges fail closed;
- import coupling uses a real parsed graph;
- normalization is stable across snapshots;
- formula/version changes are explicit; and
- history is committed or consumed by CI.

Tests must use a fixed fixture repository rather than the live source tree and
must prove candidate-set changes cannot silently rewrite historical meaning.
