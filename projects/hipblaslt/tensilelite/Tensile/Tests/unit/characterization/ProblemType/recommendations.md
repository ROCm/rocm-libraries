# Recommendations — after the `Problem.py` ProblemType slice

New file in the `ProblemType/` test dir per the add-only rule. Builds on the
`ValidParameters` "GO → Problem ProblemType slice" verdict and the cost model.

## Result for this target

The `Problem.py` ProblemType slice went from **60.07% line (601 stmts, 240
missing)** to **97.00% line (601 stmts, 18 missing; +36.93 pts)** — see
`coverage-after.txt`. The 18 residual line-misses are all provably unreachable /
dead for any GEMM `ProblemType` built through the public path (dead
post-`raise` lines, the attribute-bug `isGEMM`, GEMM-invariant index-validation
raises, a dead `UseE` `elif`, and the unreachable `Index01` ordering branch) —
line-by-line in `resistance.md`. No regression: full `-m unit` went **1563 →
1672 passed** (+109), 201 skipped unchanged.

This is the first target with a genuine dead-code residue (the prior four pure
modules hit 100%); it took the most effort (a real `__init__`/`assignDerived
Parameters`/`__str__` surface), landing within the ~1.5–2.5 day estimate.

## What worked (additions to the shared list)

- **Mirror real YAML: minimal configs, not full-default copies.** The single
  biggest correctness lever. Building `ProblemType` configs as minimal dicts
  (only the keys under test) makes the dtype-derivation `if "X" in config`
  guards behave as in production; copying `_defaultProblemType` instead pins
  `MacDataTypeA`/`DataTypeA/B` to 0 and silently defeats any `DataType` change.
- **Object-free state view via a `conftest` normaliser.** Render live
  `DataType`/`ActivationType` objects to stable strings + sort keys → snapshots
  are deterministic and review-friendly.
- **A feature-config matrix walks a branchy `__str__`.** One parametrized dict
  of ~42 named configs (dtype/HPA/transpose/bias/activation/gradient/sparse/MX/
  scale/swizzle/AllowNoFreeDims/...) covers the naming + index branches without
  bespoke tests per branch.
- **Isolate-and-restore a shared module global.** `validateProblemTypeParameter
  Types` writes Solution's collector; clear → run → capture delta → restore in
  `finally` keeps snapshots minimal and the session green.

## Go / no-go on the next target

### Verdict: **GO — `SolutionStructs/Naming.py` next** (then a small `Utilities.py` + `LdsPadding.py` top-up; defer `Solution.py` and `GlobalParameters.py`)

| Candidate | Why / why not | Effort to ≥95% line |
|---|---|---|
| **`SolutionStructs/Naming.py` (239 LOC)** ✅ chosen | **Clean imports** (`functools`, `Constants`, `RequiredParameters`, `Problem.ProblemType`); **no existing unit test**; ~11 pure string-builders (`getSolutionNameMin/Full`, `getKernelNameMin`, `getKernelFileBase`, `getParameterValueAbbreviation`, `_getName`) that turn a solution/ProblemType state into names. The natural continuation: we just pinned `ProblemType`, this names it. Pure → snapshots trivially. Mild care item: assembling a complete-enough solution `state` to drive `_getName` (reuse the LibraryIO vendored fixture's solution, or a `ProblemType` + minimal required params). | ~0.5–1 day |
| `SolutionStructs/Utilities.py` (113 LOC) + `LdsPadding.py` (412 LOC) | Both **pure** and already have partial tests (`test_SolutionStructsUtilities.py`, `test_LdsPadding.py`), so baseline is non-trivial; characterization tops them to ≥95%. `LdsPadding` is ~25 pure numeric `get_*_mt_config` padding solvers — parametrise over MT/dtype keys and snapshot. Good cheap grouped follow-up after Naming. | ~0.5–1 day each |
| `SolutionStructs/Solution.py` (5230 LOC) | The core, but a 5k-LOC monster entangled with toolchain/caps — must itself be **sliced** (e.g. `Solution`/`makeSolution`/parameter validation vs the asm-cap-driven derivation). Defer until the small pure neighbours are banked; plan a multi-slice campaign. | multi-day, sliced |
| `Common/GlobalParameters.py` (767 LOC) | Env-coupled (subprocess GPU clocks, `__version__`, mutable process globals, needs `isaInfoMap`). Monkeypatch-heavy; lower coverage-per-effort. Defer. | ~2–3 days |

**Why `Naming` now:** it is the cheapest pure module directly downstream of the
`ProblemType` we just pinned (it names solutions/kernels from that state), has
**no existing test**, and needs no toolchain. Bank it, then top up the small
pre-tested pure pair (`Utilities` + `LdsPadding`), and only then mount the
sliced `Solution.py` campaign — deferring the env-coupled `GlobalParameters`.

### Effort estimate for `Naming`

~0.5–1 day. Pure string-building; the one care item is producing a
complete-enough solution `state` for `_getName` / `getSolutionNameFull` (the
required-parameter set gates which keys appear). Reuse the LibraryIO vendored
logic fixture's solution state, or build a `ProblemType` + a minimal solution
parameter dict. A grounded API inventory + BEFORE baseline at kickoff is in the
companion `next-goal-naming.md`.
