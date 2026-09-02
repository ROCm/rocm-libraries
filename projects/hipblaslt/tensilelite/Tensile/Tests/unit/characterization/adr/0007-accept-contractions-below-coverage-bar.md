# ADR 0007: Accept `Contractions.py` below 95% combined coverage (~86%)

Status:  Accepted
Defect:  none — accepted coverage ceiling, not a bug
Commit:  b5c4440 (PR #7989) — https://github.com/ROCm/rocm-libraries/commit/b5c444007712b2c6cc7b2a3a2b3f11c8126b5921 — landed on develop via PR #7989's squash merge, 74e4693

## Context
`Contractions.py`'s index value classes, `ProblemType` (indexNames /
operationIdentifier / placeholderStr / predicates), `SizeMapping`,
`InternalArgsSupport`, and `ProblemPredicate.CompoundPredicates` are
characterized from a single vendored gfx942-HSS logic fixture. The remaining
uncovered branches are `ProblemPredicate.FromOriginalKeyPair` /
`CompoundPredicates` and `Solution` / `SizeMapping.FromOriginalState` arms
that only fire for *other* problem configurations — sparse, activation, bias
variants, batched, double/complex dtypes, GSU algorithms, and so on.
Exercising them needs a matrix of varied logic fixtures; only one is
currently vendored, and hand-authoring derived-solution states that match the
exact post-derivation serialized format is brittle and easy to get subtly
wrong.

## Decision
Characterize the surface reachable from the one vendored gfx942-HSS fixture
and accept `Contractions.py` at ~86% combined rather than hand-author
additional derived-solution states or vendor a large matrix of logic
fixtures now.

## Consequences
`Contractions.py` is a partial module, in the same spirit as the `Solution.py`
slices. A future "Contractions matrix" slice — vendoring a small,
representative set of additional logic fixtures (sparse, activation, bias,
batched, ...) — could finish coverage of the remaining predicate/state arms.

**Rejected alternatives:**
- Vendor many more logic YAMLs now — large and out of proportion to this
  module's budget.
- Hand-synthesize derived-solution states — fragile, since they must match
  the full post-derivation key set exactly; a small format drift would
  silently produce a fixture that doesn't represent real derivation output.

**Residual:** ~86% combined coverage; the uncovered ~14% is the
other-problem-configuration predicate/state arms described above.
