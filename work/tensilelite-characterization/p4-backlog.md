# P4 backlog — surviving mutants (assertion-strength gaps)

P6 mutation validation (2026-06-06, HEAD `56a34aa384b`, 6 mutants: 4 killed / 2 survived;
campaign worktree verified clean of mutation — no leak). Full report:
`coverage/p6/mutation-report.txt`. The two survivors below are lines the suite **executes**
(coverage counts them) but whose behavior change **no test asserts** — they need stronger
assertions, not more coverage. Feed back as targeted P4-style work.

## Survivor 1 — Activation Relu operand not pinned
- **Mutant:** `Tensile/Activation.py` `getReluModule` (isSingle arm), `VMaxF32(... src1=0 ...)` →
  `src1=1`. **Survived** (`test_r4_activation2` + `test_activation_char`: 180 passed).
- **Why it survived:** the activation tests assert the relu module **emits** (and which VALU op),
  but not that the clamp bound is **0** (`max(0, x)`). Changing the threshold to 1 is undetected.
- **Fix:** add an assertion that the emitted Relu instruction's `src1` operand is `0` (the clamp
  floor), per compute dtype (half/single/double/int). Mirror for LeakyRelu/ClippedRelu bounds.

## Survivor 2 — Solution auto-LocalReadVectorWidth default not pinned
- **Mutant:** `Tensile/SolutionStructs/Solution.py` `isAutoLRVW` `autoLRVW = False` → `True`.
  **Survived** (`SolutionDerivation`/`SolutionArms`/`SolutionEdges`: 98 passed).
- **Why it survived:** the auto-LRVW tests exercise the derivation path but do not pin the
  **derived LocalReadVectorWidth value** for the `LocalReadVectorWidth == -1` (auto) vs explicit
  case, so flipping the auto/non-auto default produces no asserted difference.
- **Fix:** in `SolutionDerivation`/`SolutionArms`, assert the concrete derived
  `LocalReadVectorWidth{A,B}` for an auto (`-1`) input vs an explicit input, so the auto-path
  default is pinned.

## Killed (suite catches these — no action)
- `m1` TensileMergeLibrary `fixSizeInconsistencies` size-count → KILLED.
- `m2` TensileMergeLibrary dup-trim length guard (`>= 8`) → KILLED.
- `m5` BenchmarkProblems `_cacheDataMatches` (`all(...)` → `not all(...)`) → KILLED.
- `m7` Components/StreamK `StreamKFixupTreeReduction == 1` guard → KILLED.

## Note on scope
This was a **bounded** mutation sample (6 mutants) across behaviorally-asserted code +
asm-emit code, run **strictly serially in the campaign worktree** with guaranteed revert (the
`tl-char` container is bound to this worktree, so the spec's parallel throwaway-worktree
isolation is not container-visible here — see `wf/p6-mutation.sh`). A broader sweep would
likely surface more coarse-`{basename,err}`-golden survivors in deep KWA asm generation (a
known limitation noted in `golden-governance.md`).
