# Characterization target

## Module: `Tensile/SolutionStructs/Validators/`

Three files, treated as one cohesive module (solution-level validators):

| File | Stmts | Baseline cov | Public API |
|---|---|---|---|
| `MatrixInstruction.py` | 164 | 69% | `matrixInstructionToMIParameters`, `validateMIParameters` |
| `MXScaleFormat.py` | 68 | 92% | `validateMXScaleFormatCombination` (+ `_mx*` helpers) |
| `WorkGroup.py` | 7 | 100% | `validateWorkGroup` |
| **TOTAL** | **239** | **~78%** | |

(Baseline = coverage from the *existing* unit tests, measured in the dev
container — see env README.)

## Why this module (rationale)

- **Pure-ish.** No RNG, time, filesystem, network, or module-level global
  mutable state in any of the three files (verified by grep). The only side
  effect is `reject()` writing to stdout / raising on LibraryLogic states,
  and that is gated by an explicit `printRejectionReason` argument the tests
  control. This makes deterministic snapshots achievable with no freezing.
- **Low existing coverage with headroom.** `MatrixInstruction.py` sits at
  69% — the branchy `validateMIParameters` (MFMA/WMMA/SMFMA/SWMMAC paths,
  navi vs CDNA input-per-thread, sparse, ISA remaps) is where the missing
  lines live. Good coverage-per-effort and a meaningful safety net.
- **Plausibly about to be refactored.** These validators were recently
  carved out into `Validators/` "so other Solution-level validators can
  join it" (per the module docstring), and `validateMIParameters` reaches
  into `Solution.py`-shaped dicts with overlapping logic to
  `matrixInstructionToMIParameters` — a classic candidate for consolidation.
  Pinning current behaviour first de-risks that refactor.
- **Structured output suits snapshots.** `matrixInstructionToMIParameters`
  returns a ~25-key MI-parameter dict; the `validate*` fns return a bool +
  mutate `state["Valid"]`. Snapshotting the structured return (not raw
  blobs) is exactly the goal's intent.
- **Self-contained inputs.** Inputs are plain dicts/lists plus an
  `isaInfoMap` (ISA→caps). No GPU and no C++ client are needed (the
  `unit`/`coverage-unit` path), so the suite runs CPU-only.

## Deviation from the goal's stated path

The goal text names `tests/characterization/<target>/`. That path is NOT in
`testpaths` (`Tensile/Tests rocisa/test`) and sits beside the C++ gtests in
`tests/`, so the default `pytest` would not collect it and the ≥95% `--cov`
gate would need a bespoke invocation. To satisfy *both* "≥95% via
`pytest --cov`" and "no regression / same pytest invocation", the suite
lives at:

```
Tensile/Tests/unit/characterization/Validators/
```

marked `-m unit`, so the existing default invocation and the
`coverage-unit`/`unit` tox envs pick it up unchanged.

## Coverage measurement (the one rule that matters)

Run from the tensilelite dir; pass `--cov` a **directory path** (not a
dotted module name) so coverage scans rather than imports — importing a
rocisa-touching module re-inits the `_rocisa` nanobind extension and aborts
(SIGABRT). See env README for the full root-cause.

```
pytest -m unit \
  --cov=Tensile/SolutionStructs/Validators \
  --cov-config=pyproject.toml --cov-report=term-missing \
  Tensile/Tests/unit
```
