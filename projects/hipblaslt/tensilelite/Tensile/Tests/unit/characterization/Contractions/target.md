# Characterization target — `Tensile/Contractions.py`

Part of the master-plan remaining-module sweep. **Before 84.2% → after ~86%
combined**. Covers FreeIndex/BatchIndex/BoundIndex, ProblemType (indexNames /
operationIdentifier / placeholderStr / predicates), SizeMapping /
InternalArgsSupport / ProblemPredicate.CompoundPredicates, driven from the
vendored gfx942-HSS logic fixture (raw problemType + a fully-derived solution
state from parsing). 8 tests.

**Accepted <95% — see DECISIONS D10.** The residual is the predicate-generation
+ FromOriginalState arms that only fire for other problem configs (sparse /
activation / bias / batched / double-complex / GSU variants) — a
varied-logic-fixture MATRIX this sweep doesn't have.
