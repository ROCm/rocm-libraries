# Recommendations — after `Tensile/SolutionStructs/Naming.py`

New file in the `Naming/` test dir per the add-only rule. Builds on the
`ProblemType` "GO → Naming next" verdict and the running cost model.

## Result for this target

`Naming.py` went from **82.50% line (120 stmts, 21 missing)** to **99.17% line
(120 stmts, 1 missing; +16.67 pts)** — see `coverage-after.txt`. The 1 residual
line is a provably-unreachable defensive `raise`. The suite also **characterized
a latent bug** — `getKernelNameMin(splitGSU=True)` with `GlobalSplitU > 1` or
`-1` raises `TypeError` (`"M" > 0` at L160), pinned via `pytest.raises` — see
`resistance.md`. No regression: full `-m unit` **1672 → 1713 passed** (+41),
201 skipped unchanged.

Came in under the ~0.5–1 day estimate (a few hours): pure string-building, one
`conftest` state factory, one test file.

## What worked (additions to the shared list)

- **A single state-factory fixture for name builders.** `make_state` assembles
  the minimum a name builder reads — a real `ProblemType` + `GlobalSplitU` +
  the 9 internal-args keys + `SpaceFillingAlgo` + tile keys — overridable per
  test. The two KeyErrors that bit first (`SpaceFillingAlgo` under the Full
  required set; the internal-args under `getKeyNoInternalArgs`) became one-line
  factory additions.
- **Characterize bugs, don't fix them.** The `"M" > 0` `TypeError` is real
  current behaviour; pinning it via `pytest.raises` (rather than skipping or
  fixing, which add-only forbids) both documents the defect and makes a future
  fix surface as an intentional expectation change.
- **Pin mutate-then-restore contracts.** `_getName`/`getKeyNoInternalArgs`
  temporarily rewrite `GlobalSplitU`/`GroupedGemm`; explicit "no-mutate" tests
  pin the exact restore.

## Go / no-go on the next target

### Verdict: **GO — `SolutionStructs/Utilities.py` + `LdsPadding.py` top-up** (one grouped target; then the sliced `Solution.py` campaign; defer `GlobalParameters.py`)

| Candidate | Current | Why / why not | Effort to ≥95% line |
|---|---|---|---|
| **`SolutionStructs/Utilities.py` (49 stmts, 46.84%)** ✅ | low | Small and **pure**: `getMiInputType`, `reject`, `pvar`, `roundupRatio`, `getRealDataType{A,B}`. The existing `test_SolutionStructsUtilities.py` barely touches it; snapshot the dtype/reject/round branches. Highest coverage-per-effort left. | ~0.25–0.5 day |
| **`SolutionStructs/LdsPadding.py` (212 stmts, 86.45%)** ✅ | high | **Pure numeric** padding solvers (`get_fp4/fp8/fp16/fp32/mxs_mt_config` + `_compute_*`/`_check`/`_search_padding`). Already well-covered by `test_LdsPadding.py`; a characterization parametrised over MT / dtype-key / wave params tops it to ≥95% and pins the padding tables. Pairs naturally with Utilities as one pure-`SolutionStructs` top-up suite. | ~0.5 day |
| `Common/GlobalParameters.py` (220 stmts, 84.19%) | high | Already decent but **env-coupled** (subprocess GPU clocks, `__version__`, mutable process globals, needs `isaInfoMap`). A monkeypatch-heavy top-up; lower priority than the pure pair. | ~1–1.5 day |
| `SolutionStructs/Solution.py` (3272 stmts, 32.78%) | low | The core, but a **5230-LOC monster** entangled with toolchain/caps. Must be a **multi-slice campaign** (e.g. parameter validation → `Solution` assembly → asm-cap-driven derivation), each slice its own target. Mount after the cheap neighbours are banked. | multi-day, sliced |

**Why the pure pair now:** `Utilities` is the single cheapest remaining module
(small, pure, currently <50%), and `LdsPadding` is a pure numeric top-up that
shares the `SolutionStructs` neighbourhood — together they're a ~1-day grouped
target that finishes the pure `SolutionStructs` surface before the big
`Solution.py` campaign. Defer the env-coupled `GlobalParameters` and the sliced
`Solution.py` to after.

### Effort estimate

~0.5–1 day combined. Both pure, no toolchain/GPU. `LdsPadding` needs a small
matrix of (MT, key, wave/vw) inputs; `Utilities.reject` mutates `state` and
prints — snapshot the return + the reject-reason effect with
`printSolutionRejectionReason=False` for determinism (as the `Validators` suite
did). A grounded API inventory + BEFORE baseline at kickoff is in the companion
`next-goal-solutionstructs-utils.md`.
