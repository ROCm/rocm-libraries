# ADR 0003: Pickle-free `.ambr` derived-state golden as the kill vehicle for the `assignDerivedParameters` giant

Status:  Accepted
Defect:  n/a (mutation-hardening, add-only test)

## Context
`Solution.assignDerivedParameters` is the largest mutable surface in the
TensileLite Python tree (the `depthU` and `adp` mutant families alone are ~18820
mutants). Its derivation cannot be exercised by a handful of per-mutant kill
tests: a mutation of any single derived field must be observable, which requires
asserting the *complete* derived `Solution._state`, not a fixed column subset.

The interim kill corpus used a committed pickle of pre-derive states. A pickle is
opaque in review, couples the golden to a Python/pickle-protocol version, and is
not diffable. The characterization suite's discipline is diffable, add-only
goldens.

## Decision
Introduce `SolutionDerivation/test_derivation_golden_char.py` with a syrupy
`.ambr` golden (`__snapshots__/test_derivation_golden_char.ambr`, 246 cases) as
the kill vehicle, and commit **no pickle**. Input cases are regenerated at
collection time from 59 in-tree designed base YAML configs under
`_codegen/data/test_data/_designed` by replaying the exact corpus pipeline
(BenchmarkProcess -> constructForkPermutations capped at LIMIT ->
`_generate_single_solution` with a monkeypatch hook capturing a deepcopy of the
pre-derive state), then calling `assignDerivedParameters` in the test body. The
full state is rendered address-free; nested Mappings (e.g. `ProblemType`) recurse
so every field is deep-compared exactly as the pickle golden did.

Two kill-power gaps found while establishing equivalence were fixed in the test
harness (not in source): derivation was moved out of the collection phase (a
collection-time `try/except` was swallowing mutants that raise), and `_sanitize`
now recurses into any `Mapping` so `ProblemType` is deep-compared (closing 4
`MirrorDimsMetadata` mutants that a `str()`-only render missed).

## Consequences
The golden is byte-reproducible for a fixed toolchain/container (verified: two
further runs with no `--snapshot-update` are byte-identical) and diffable in
review. It was verified kill-equivalent to the interim pickle corpus over 8
stratified windows across `assignDerivedParameters` (lines 1567-2857, 684
mutants, 0 per-key exit-code divergence), so dropping the pickle costs no kill
power.

Reproduction depends on a fixed toolchain/container: this golden pins host-side
Python derivation, not codegen digests, so it is a derivation-regression net, not
a cross-arch codegen-stability signal. If the designed configs or the derivation
pipeline change intentionally, regenerate with `--snapshot-update`, re-run twice
to confirm byte-stability, and record the regeneration in `DECISIONS.md`.
