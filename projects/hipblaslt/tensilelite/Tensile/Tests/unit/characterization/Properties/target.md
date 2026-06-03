# Characterization target — `Tensile/Properties.py`

Part of the master-plan remaining-module sweep. **Before 80.7% → after 100.00%
line** (57 stmts, 0 miss; 97.18% blended). Drives `Property` (state/eq/hash/
FromOriginalState) and `Predicate` (And/Or for 0/1/2+ predicates; `__lt__`
matching-order + default + dict-value paths). Residual: 2 partial branches in
`__lt__` (106->110, 110->115 — the dict-vs-dict reduction only triggers when
*both* operands carry dict values; line coverage is 100%).
